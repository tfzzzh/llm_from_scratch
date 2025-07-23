import torch
import triton
import triton.language as tl
import math


class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        BLOCK_Q = 16
        BLOCK_K = 16
        bsize, seqlen_q, dim = Q.shape
        _, seqlen_k, _ = K.shape

        # allocate outputs
        O = torch.zeros_like(Q)
        L = torch.zeros(bsize, seqlen_q, dtype=Q.dtype, device=Q.device)

        with torch.no_grad():
            for b in range(bsize):
                Ob, Lb = FlashAttentionPytorch._compute_one(Q[b], K[b], V[b], is_causal, BLOCK_Q, BLOCK_K)
                O[b] = Ob
                L[b] = Lb
        
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.BLOCK_Q = BLOCK_Q
        ctx.BLOCK_K = BLOCK_K
        return O
    
    @staticmethod
    def _compute_one(Q, K, V, is_causal, BLOCK_Q, BLOCK_K):
        seqlen_q, dim = Q.shape
        seqlen_k, _ = K.shape

        assert seqlen_q % BLOCK_Q == 0 and seqlen_k % BLOCK_K == 0
        num_block_q = seqlen_q // BLOCK_Q
        num_block_k = seqlen_k // BLOCK_K

        Os = []
        ls = []
        for i in range(num_block_q):
            Qi = Q[i * BLOCK_Q : (i+1) * BLOCK_Q]
            Oi = torch.zeros_like(Qi)
            li = torch.zeros(BLOCK_Q, device=Q.device, dtype=Q.dtype)
            mi = torch.empty(BLOCK_Q, device=Q.device, dtype=Q.dtype)
            mi_old = torch.zeros_like(mi)
            mi[:] = float('-inf')

            for j in range(num_block_k):
                Kj = K[j * BLOCK_K : (j+1) * BLOCK_K]
                Vj = V[j * BLOCK_K : (j+1) * BLOCK_K]
                Sij = Qi @ (Kj.T) / math.sqrt(dim)
                mi_old[:] = mi
                mi[:] = torch.max(mi, torch.max(Sij, dim=1).values)
                Pij = torch.exp(Sij - mi[:,None])
                
                # mi_old - mi
                delta = mi_old - mi
                reweight = torch.exp(delta)
                assert torch.isnan(reweight).sum() == 0

                li[:] = reweight * li + Pij.sum(dim=1)
                Oi[:] = reweight[:,None] * Oi + Pij @ Vj

            Oi[:] = (1.0 / li)[:, None] * Oi
            li[:] = mi + torch.log(li)
            Os.append(Oi)
            ls.append(li)

        # concate Os
        return torch.concat(Os, dim=0), torch.concat(ls, dim=0)
    

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        bsize, nseq_q, dim = Q.shape
        _, nseq_k, ndim = K.shape
        BLOCK_K = ctx.BLOCK_K
        BLOCK_Q = ctx.BLOCK_Q
        assert nseq_k % BLOCK_K == 0 and nseq_q % BLOCK_Q == 0

        Tk = nseq_k // ctx.BLOCK_K
        Tq = nseq_q // ctx.BLOCK_Q

        # init output
        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        # D = rowsum(dO * O)
        D = torch.sum(dO * O, dim=-1) # [bsize, nseq_q]

        for b in range(bsize):
            for j in range(Tk):
                # Load K_j = K[b, jth-block, :] && Vj = V[b, jth-block, :]
                Kj = K[b, j*BLOCK_K : (j+1) * BLOCK_K, :]
                Vj = V[b, j*BLOCK_K : (j+1) * BLOCK_K, :]
                dKj = torch.zeros_like(Kj)
                dVj = torch.zeros_like(Vj)

                for i in range(Tq):
                    # Load Qi = Q[b, i-th block, :]
                    # Load Oi, dOi, Li using same layer out
                    Qi = Q[b, i*BLOCK_Q : (i+1) * BLOCK_Q, :]
                    # Oi = O[b, i*BLOCK_Q : (i+1) * BLOCK_Q, :]
                    dOi = dO[b, i*BLOCK_Q : (i+1) * BLOCK_Q, :]
                    Li = L[b, i*BLOCK_Q : (i+1) * BLOCK_Q]
                    Di = D[b, i*BLOCK_Q : (i+1) * BLOCK_Q]
                    # dQi = dQ[b, i*BLOCK_Q : (i+1) * BLOCK_Q, :]

                    Sij = (Qi @ Kj.T) / math.sqrt(ndim) # shape: [BQ, BK]
                    Pij = torch.exp(Sij - Li[:, None]) 

                    dVj += Pij.T @ dOi
                    dPij = dOi @ Vj.T 
                    dSij = Pij * (dPij - Di[:, None]) / math.sqrt(ndim)

                    dQ[b, i*BLOCK_Q : (i+1) * BLOCK_Q, :] += dSij @ Kj
                    dKj += dSij.T @ Qi

                #store
                dK[b, j*BLOCK_K : (j+1) * BLOCK_K, :] = dKj
                dV[b, j*BLOCK_K : (j+1) * BLOCK_K, :] = dVj
        
        return dQ, dK, dV, None

# [TODO]: support casual
# [TODO]: support float16
# Assumption Q: stride(2) < sride(1)
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    casual: tl.constexpr
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    # use to load Q[b, qidx * Q_TILE_SIZE  : (qidx+1) * Q_TILE_SIZE, :]
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # Use K[b, :, :]
    # first load block K[b, 0: K_TILE_SIZE, :]
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape = (N_KEYS, D),
        strides = (stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    # Use V[b, :, :]
    # first load block V[b,  0: K_TILE_SIZE, :]
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape = (N_KEYS, D),
        strides = (stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    # Use O[b, qidx * Q_TILE_SIZE : (qidx+1) * Q_TILE_SIZE, :]
    # first block location: U[b, qidx * Q_TILE_SIZE, 0]
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    # Use L[b, qidx * Q_TILE_SIZE : (qidx+1) * Q_TILE_SIZE]
    # first block: L[b, qidx * Q_TILE_SIZE]
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    # Load Qi from global memory
    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    Oi = tl.zeros(shape=(Q_TILE_SIZE, D), dtype=tl.float32)
    Li = tl.zeros(shape=(Q_TILE_SIZE,), dtype=tl.float32)
    mi = tl.zeros(shape=(Q_TILE_SIZE,), dtype=tl.float32) + float('-inf')
    mi_prev = tl.zeros(shape=(Q_TILE_SIZE,), dtype=tl.float32)

    # compute mask of i (arange only support tl.constexpr)
    q_row_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    q_row_min = query_tile_index * Q_TILE_SIZE # least casual position of current q tile
    # q_row_mask = q_row_range < N_QUERIES

    tk = tl.cdiv(N_KEYS, K_TILE_SIZE)

    if casual:
        # For causal attention, we only need to process up to the current query position
        # j * K_TILE_SIZE <= q_row_min
        # 
        # max_k_tile = tl.cdiv(q_row_min + Q_TILE_SIZE, K_TILE_SIZE) 
        max_k_tile = q_row_min // K_TILE_SIZE + 1
        # max_k_tile = tl.cdiv(q_row_min+1, K_TILE_SIZE)
        tk = tl.minimum(tk, max_k_tile)

    # how to handle when N_KEYS % K_TILE_SIZE not 0?
    # mask out in valid part of Sij to -inf
    # special case: what if a row of si,j being infty
    for j in range(tk):
        # load K[b, j*K_TILE_SIZE : (j+1)*K_TILE_SIZE, :]
        # load V[b, j*K_TILE_SIZE : (j+1)*K_TILE_SIZE, :]
        # k_offset >= j * K_TILE_SIZE
        # k_col_min = j * K_TILE_SIZE # least casual position of current key tile
        # if casual and k_col_min > q_row_min: break

        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option='zero')
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option='zero')
        Sij = tl.dot(Qi, Kj.T) / scale

        # get valid mask for j
        k_col_offsets = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        k_col_mask = k_col_offsets < N_KEYS 

        if casual:
            casual_mask = k_col_offsets[None, :] <= q_row_offsets[:,None]
            casual_mask = casual_mask & (k_col_mask[None, :])

        # set score of masked out column to -float
        if casual:
            Sij = Sij + tl.where(casual_mask, 0.0, float('-inf'))
        else:
            Sij = Sij + tl.where(k_col_mask, 0.0, float('-inf'))[None, :]

        # update mi
        mi_prev = mi
        mi = tl.maximum(mi, tl.max(Sij, axis=1))
        Pij = tl.exp(Sij - tl.expand_dims(mi, axis=1))
        reweight = tl.exp(mi_prev - mi)
        Li = reweight * Li + tl.sum(Pij, axis=1)
        Oi = tl.expand_dims(reweight, axis=1) * Oi # + tl.dot(Pij, Vj)
        Oi = tl.dot(Pij, Vj, Oi) # error: tl.dot(Pij, Vj, Oi)

        # advance k, v pointers
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

        # if j == tk-1:
        #     tl.device_print("Oi=", Oi)
    
    Oi = tl.expand_dims(1.0 / Li, axis=1) * Oi
    Li = mi + tl.log(Li)

    # store to block
    tl.store(O_block_ptr, Oi, boundary_check=(0,1))
    tl.store(L_block_ptr, Li, boundary_check=(0,))


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        BLOCK_Q = 16
        BLOCK_K = 16
        bsize, seqlen_q, dim = Q.shape
        _, seqlen_k, _ = K.shape

        # allocate outputs
        O = torch.zeros_like(Q)
        L = torch.zeros(bsize, seqlen_q, dtype=Q.dtype, device=Q.device)
        
        flash_fwd_kernel[(triton.cdiv(seqlen_q, BLOCK_Q), bsize)](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            seqlen_q, seqlen_k,
            math.sqrt(dim),
            dim,
            BLOCK_Q,
            BLOCK_K,
            is_causal
        )

        # save ctx
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.BLOCK_Q = BLOCK_Q
        ctx.BLOCK_K = BLOCK_K
        ctx.is_causal = is_causal
        return O
    
    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        bsize, nseq_q, ndim = Q.shape
        _, nseq_k, _ = K.shape
        BLOCK_K = ctx.BLOCK_K
        BLOCK_Q = ctx.BLOCK_Q
        is_causal = ctx.is_causal

        Tk = triton.cdiv(nseq_k, ctx.BLOCK_K)
        Tq = triton.cdiv(nseq_q, ctx.BLOCK_Q)

        # init output
        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        # D = rowsum(dO * O)
        D = torch.sum(dO * O, dim=-1) # [bsize, nseq_q]

        # partition computation by (k_tile, batch_size)
        flash_bwd_dkv_kernel[(Tk,bsize)](
            Q, K, V,
            O, L,
            dO, D,
            dQ, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            dO.stride(0), dO.stride(1), dO.stride(2),
            D.stride(0), D.stride(1),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            nseq_q, nseq_k,
            1.0 / math.sqrt(ndim),
            ndim,
            BLOCK_Q,
            BLOCK_K,
            is_causal
        )

        flash_bwd_dq_kernel[(Tq,bsize)](
            Q, K, V,
            O, L,
            dO, D,
            dQ, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            dO.stride(0), dO.stride(1), dO.stride(2),
            D.stride(0), D.stride(1),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            nseq_q, nseq_k,
            1.0 / math.sqrt(ndim),
            ndim,
            BLOCK_Q,
            BLOCK_K,
            is_causal
        )

        return dQ, dK, dV, None

# TODO handle casual
@triton.jit
def flash_bwd_dkv_kernel(
    # inputs
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    dO_ptr, D_ptr,
    # output
    dQ_ptr, dK_ptr, dV_ptr,
    # strides
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_dob, stride_doq, stride_dod,
    stride_db, stride_dq,
    stride_dqb, stride_dqq, stride_dqd,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS,
    scale, # 1 / sqrt(d)
    DIM: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    casual: tl.constexpr
):
    # Program indices
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    # use to load Q[b, :, :]
    # first load Q[b, 0-th block, :]
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, DIM),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, DIM),
        order=(1, 0),
    )

    # Use K[b, key_tile_index-th block, :]
    # first load block K[b, key_tile_index * K_TILE_SIZE, :]
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape = (N_KEYS, DIM),
        strides = (stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, DIM),
        order=(1, 0)
    )

    # Use V[b, key_tile_index-th block, :]
    # first load block addr V[b,  key_tile_index * K_TILE_SIZE, 0]
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape = (N_KEYS, DIM),
        strides = (stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, DIM),
        order=(1, 0)
    )

    # Use L[b, :]
    # first block: L[b, 0]
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    # Use dO[b, :, :]
    # first addr dO[b, 0, 0]
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, DIM),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, DIM),
        order=(1, 0),
    )

    # use D[b, :]
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    # use dQ[b, :, :]
    # dQ_block_ptr = tl.make_block_ptr(
    #     dQ_ptr + batch_index * stride_dqb,
    #     shape=(N_QUERIES, DIM),
    #     strides=(stride_dqq, stride_dqd),
    #     offsets=(0, 0),
    #     block_shape=(Q_TILE_SIZE, DIM),
    #     order=(1, 0),
    # )

    # use dK[b, key_idx-th block, :]
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape = (N_KEYS, DIM),
        strides = (stride_dkk, stride_dkd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, DIM),
        order=(1, 0)
    )

    # use dV[b, key-idx-th block, :]
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape = (N_KEYS, DIM),
        strides = (stride_dvk, stride_dvd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, DIM),
        order=(1, 0)
    )

    # Load K_j = K[b, jth-block, :] && Vj = V[b, jth-block, :]
    # Kj is of shape [K_TILE_SIZE, D]
    Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
    dKj = tl.zeros(shape=(K_TILE_SIZE, DIM), dtype=tl.float32)
    dVj = tl.zeros(shape=(K_TILE_SIZE, DIM), dtype=tl.float32)
    offset_j = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)

    Tq = tl.cdiv(N_QUERIES, Q_TILE_SIZE)
    for i in range(0, Tq):
        # Load Qi = Q[b, i-th block, :], dOi, Li, Di
        Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        dOi =  tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
        Li = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        Di = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")

        offset_i = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        mask = (offset_i[:, None] < N_QUERIES) & (offset_j[None, :] < N_KEYS)
        if casual:
            mask_casual = offset_j[None, :] <= offset_i[:, None]
            mask = mask & mask_casual

        Sij = tl.dot(Qi, Kj.T) * scale
        Pij = tl.exp(Sij - Li[:, None])
        Pij = tl.where(mask, Pij, 0.0)

        dVj += tl.dot(Pij.T, dOi)
        dPij = tl.dot(dOi, Vj.T)
        dSij = Pij * (dPij - Di[:, None]) * scale
        # dQi = tl.load(dQ_block_ptr, boundary_check=(0, 1), padding_option="zero")
        # dQi = tl.dot(dSij, Kj)
        # tl.store(dQ_block_ptr, dQi, boundary_check=(0, 1))
        # tl.atomic_add(dQ_block_ptr, dQi, )

        dKj += tl.dot(dSij.T, Qi)

        # update pointer of Q, dO, L, D, dQ
        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        # dQ_block_ptr = dQ_block_ptr.advance((Q_TILE_SIZE, 0))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))

    # write dKj, dVj back
    tl.store(dK_block_ptr, dKj, boundary_check=(0, 1))
    tl.store(dV_block_ptr, dVj, boundary_check=(0, 1))


@triton.jit
def flash_bwd_dq_kernel(
    # inputs
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    dO_ptr, D_ptr,
    # output
    dQ_ptr, dK_ptr, dV_ptr,
    # strides
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_dob, stride_doq, stride_dod,
    stride_db, stride_dq,
    stride_dqb, stride_dqq, stride_dqd,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS,
    scale, # 1 / sqrt(d)
    DIM: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    casual: tl.constexpr
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    # use to load Q[b, query_tile_index-th block, :]
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, DIM),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, DIM),
        order=(1, 0),
    )

    # Use K[b, :, :]
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape = (N_KEYS, DIM),
        strides = (stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, DIM),
        order=(1, 0)
    )

    # Use V[b, :, :]
    # first load block addr V[b,  key_tile_index * K_TILE_SIZE, 0]
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape = (N_KEYS, DIM),
        strides = (stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, DIM),
        order=(1, 0)
    )

    # Use L[b, query_tile_index-block]
    # first block: L[b, 0]
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    # Use dO[b, q-th block, :]
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, DIM),
        strides=(stride_doq, stride_dod),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, DIM),
        order=(1, 0),
    )

    # use D[b, q-th block]
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    # use dQ[b, q-th block, :]
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, DIM),
        strides=(stride_dqq, stride_dqd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, DIM),
        order=(1, 0),
    )

    Tk = tl.cdiv(N_KEYS, K_TILE_SIZE)

    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    dOi =  tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
    Li = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
    Di = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")
    dQi = tl.zeros(shape=(Q_TILE_SIZE, DIM), dtype=tl.float32)
    offset_i = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    for j in range(Tk):
        # load data
        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        offset_j = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        mask = (offset_i[:, None] < N_QUERIES) & (offset_j[None, :] < N_KEYS)
        if casual:
            mask_casual = offset_j[None, :] <= offset_i[:, None]
            mask = mask & mask_casual

        Sij = tl.dot(Qi, Kj.T) * scale
        Pij = tl.exp(Sij - Li[:, None])
        Pij = tl.where(mask, Pij, 0.0)
        dPij = tl.dot(dOi, Vj.T)
        dSij = Pij * (dPij - Di[:, None]) * scale
        dQi += tl.dot(dSij, Kj)

        # update pointer
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE,0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE,0))
    
    tl.store(dQ_block_ptr, dQi, boundary_check=(0, 1))


def _scaled_dot_product_flash_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    is_casual: bool = True
) -> torch.Tensor:
    """scaled dot-product attention function, only casual mask is supported

    Args:
        Q (torch.Tensor): shape (batch_size, ..., seq_len_q, d_q)
        K (torch.Tensor): shape (batch_size, ..., seq_len_k, d_k)
        V (torch.Tensor): shape (batch_size, ..., seq_len_k, d_v)
        mask (torch.Tensor): shape (batch_size, ..., seq_len_q, seq_len_k), when set to true attention applied

    Returns:
        torch.Tensor: _description_
    """
    # make input tensor being 4-d
    assert Q.ndim == K.ndim and K.ndim == V.ndim
    assert Q.dtype == torch.float32, 'only float32 is supported'

    if Q.ndim == 3:
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()
        out = FlashAttentionTriton.apply(Q, K, V, is_casual)
    
    else:
        bsize, num_head, seq_len_q, dim = Q.shape
        _, _, seq_len_k, _ = K.shape

        Q = Q.reshape(bsize * num_head, seq_len_q, dim).contiguous()
        K = K.reshape(bsize * num_head, seq_len_k, dim).contiguous()
        V = V.reshape(bsize * num_head, seq_len_k, dim).contiguous()

        out = FlashAttentionTriton.apply(Q, K, V, is_casual)
        out = out.reshape(bsize, num_head, seq_len_q, dim)

    return out
