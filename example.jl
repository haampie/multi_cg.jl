using LinearAlgebra

import LinearAlgebra: mul!, ldiv!
import Base: size
using Base: OneTo

"""
This is the linear operator
H(X) = A * X - B * X * D + γ * P * (P' * (B * X)
where D = Diagonal(λs)
products are evaluated as follows:
C ← α * A * B + β * C
tmp = D * B
C ← -α * S * tmp + 1 * C
"""
struct MatrixOp{Tv,TM<:AbstractMatrix{Tv},Tλ} <: AbstractMatrix{Tv}
    A::TM
    B::TM
    P::TM
    λs::Tλ
end

Base.size(A::MatrixOp) = size(A.A)

struct PrecOp{Tv,TM<:AbstractVector{Tv}} <: AbstractMatrix{Tv}
    A_diag::TM
    B_diag::TM
    λs::TM
end

Base.size(P::PrecOp) = (length(P.A_diag), length(P.A_diag))

function ldiv!(P::PrecOp, x::AbstractVecOrMat)
    for j in OneTo(size(x, 2))
        for i in OneTo(size(x, 1))
            # make sure not to divide by 0
            # x[i, j] /= max(1.0, )
            val = P.A_diag[i] - P.λs[j] * P.B_diag[i]
            x[i, j] /= sqrt(val^2 + 0.01)
        end
    end
end

function the_mul!(Y::AbstractVecOrMat, H::MatrixOp, X::AbstractVecOrMat, α::Number, β::Number)
    mul!(Y, H.A, X, α, β)
    tmp = X * Diagonal(view(H.λs, 1:size(X, 2)))
    mul!(Y, H.B, tmp, -α, 1)
    Y .+= (last(H.λs) + 1) .* (H.P * (H.P' * (H.B * X)))
    return Y
end

function repack!(x::AbstractVector, ids)
    for (i, j) in enumerate(ids)
        if i != j
            x[i] = x[j]
        end
    end
    return nothing
end

repack!(M::MatrixOp, ids) = repack!(M.λs, ids)
repack!(::UniformScaling, ::AbstractVector) = nothing
repack!(P::PrecOp, ids) = repack!(P.λs, ids)

function repack!(X::AbstractMatrix, ids)
    for (i, j) in enumerate(ids)
        if i != j
            copyto!(view(X, :, i), view(X, :, j))
        end
    end
    return nothing
end

mul!(C::AbstractVector, H::MatrixOp, B::AbstractVector, α::Number, β::Number) = the_mul!(C, H, B, α, β)
mul!(C::AbstractMatrix, H::MatrixOp, B::AbstractMatrix, α::Number, β::Number) = the_mul!(C, H, B, α, β)

function block_cg!(X, A, P, B, U, C; maxiters = 10, tol=1e-6)
    m, n = size(X)

    @assert m ≥ n
    @assert size(X, 1) == size(B, 1) == size(U, 1) == size(C, 1)
    @assert size(X, 2) == size(B, 2) == size(U, 2) == size(C, 2)

    fill!(U, 0)

    # Use B effectively as the residual block-vector
    # B = B - A * X
    mul!(B, A, X, -1.0, 1.0)

    rs = zeros(n)
    ρs = ones(n)
    ρs_old = ones(n)
    σs = zeros(n)

    # When vectors converge we move them to the front, but we can't really do
    # that with X, so we have to keep track of where is what.
    ids = collect(1:n)

    num_unconverged = n

    residual_history = [Float64[] for _ = 1:n]

    for iter = 1:maxiters
        # Check the residual norms.
        # we could also check ρs, which is the residual norm in the
        # P⁻¹-inner product, but if P is nearly singular it might not be a
        # great norm.
        for i = 1:num_unconverged
            rs[i] = norm(view(B, :, i))

            push!(residual_history[ids[i]], rs[i])
        end

        not_converged = [i for (i, v) in enumerate(view(rs, OneTo(num_unconverged))) if v > tol]

        num_unconverged = length(not_converged)

        if length(not_converged) == 0
            break
        end

        # Todo, repacking
        repack!(ids, not_converged)
        repack!(U, not_converged)
        repack!(B, not_converged)
        repack!(A, not_converged)
        repack!(P, not_converged)
        repack!(ρs, not_converged)

        ldiv!(view(C, :, OneTo(num_unconverged)), P, view(B, :, OneTo(num_unconverged)))

        ρs_old[OneTo(num_unconverged)] .= view(ρs, OneTo(num_unconverged))

        for i = 1:num_unconverged
            ρs[i] = dot(view(C, :, i), view(B, :, i))
        end

        # In the first iteration we have U == 0, so no need for an axpy.
        if iter == 1
            copyto!(view(U, :, OneTo(num_unconverged)), view(C, :, OneTo(num_unconverged)))
        else
            for i = 1:num_unconverged
                axpby!(1.0, view(C, :, i), ρs[i] / ρs_old[i], view(U, :, i))
            end
        end

        mul!(view(C, :, OneTo(num_unconverged)), A, view(U, :, OneTo(num_unconverged)))

        for i = 1:num_unconverged
            σs[i] = dot(view(U, :, i), view(C, :, i))
        end

        for i = 1:num_unconverged
            α = ρs[i] / σs[i]
            axpy!(α, view(U, :, i), view(X, :, ids[i]))
            axpy!(-α, view(C, :, i), view(B, :, i))
        end
    end

    return X, residual_history
end

function rand_spd(m, λs = range(1.0, m, length=m))
    F = qr!(rand(m, m))
    A = F.Q * Diagonal(λs) * F.Q'
    return A
end

function example(m, n)
    # random SPD matrix
    A = rand_spd(m)

    # identity as preconditioner
    P = 1.0I

    # True solution
    X_true = rand(m, n)

    # Right-hand side
    B = A * X_true

    # CG destroys the right-hand side, so make a copy.
    B_cg = copy(B)

    # Initial guess
    X_cg = rand(m, n)

    # Auxilary matrices
    U, C = zeros(m, n), zeros(m, n)

    block_cg!(X_cg, A, P, B_cg, U, C)

end

function bigger_example(m, n; maxiters = 50)
    A, B = rand_spd(m), rand_spd(m, ones(m))
    λs, Q = eigen(A, B)

    # 𝓟 = P * P' * B is a projector onto the first m eigenvectors.
    # 𝓟 * 𝓟 = P * (P' * B * P) * P' * B = P * P' * B = 𝓟
    # and so I - 𝓟 = I - P * P' * B projects on the rest of the spectrum.
    P = Q[:, 1:n]

    H = MatrixOp(A, B, P, λs[1:n])

    # Create a true solution, and apply I - 𝓟 to remove a bunch of eigenvecs.
    X_true = rand(m, n)
    X_true .-= P * (P' * (B * X_true))
    X_true .-= P * (P' * (B * X_true)) # apply it twice to avoid rounding errors

    rhs = H * X_true
    rhs .-= P * (P' * (B * rhs))
    rhs .-= P * (P' * (B * rhs))

    # The right-hand side should be orthogonal to P
    @show norm(P' * (B * rhs))
    @show norm(H * X_true - rhs)
    
    # Allocate a bunch of things to run block cg.
    
    prec = 1.0 * I # <- use an identity preconditioner

    # initial guess
    X_cg = rand(m, n)
    X_cg .-= P * (P' * (B * X_cg))
    X_cg .-= P * (P' * (B * X_cg))

    # cg will destroy the B blockvector
    R = copy(rhs)

    # Auxilary stuff
    U, C = zeros(m, n), zeros(m, n)

    block_cg!(X_cg, H, prec, R, U, C, maxiters=maxiters)

    @show norm(H * X_cg - rhs)
    @show norm(X_cg - X_true)
end

function standard_evp(m, n; maxiters = 50, reltol=1e-7)
    A, B = rand_spd(m), Matrix(1.0I, m, m)
    λs, Q = eigen(A)

    # 𝓟 = P * P' is a projector onto the first m eigenvectors.
    # 𝓟 * 𝓟 = P * (P' * P) * P' = P * P' = 𝓟
    # and so I - 𝓟 = I - P * P' projects on the rest of the spectrum.
    P = Q[:, 1:n]

    H = MatrixOp(A, B, P, λs[1:n])

    # Create a true solution, and apply I - 𝓟 to remove a bunch of eigenvecs.
    X_true = rand(m, n)
    X_true .-= P * (P' * X_true)
    X_true .-= P * (P' * X_true) # apply it twice to avoid rounding errors

    rhs = H * X_true

    # The right-hand side should be orthogonal to P
    @show norm(P' * rhs)
    
    # Allocate a bunch of things to run block cg.
    prec = 1.0 * I # <- use an identity preconditioner
    # prec = PrecOp(diag(A), diag(B), λs[1:n])

    # initial guess
    X_cg = rand(m, n)
    X_cg .-= P * (P' * X_cg)
    X_cg .-= P * (P' * X_cg)

    # cg will destroy the B blockvector
    R = copy(rhs)

    # Auxilary stuff
    U, C = zeros(m, n), zeros(m, n)

    _, hist = block_cg!(X_cg, H, prec, R, U, C, maxiters=maxiters, tol=reltol)

    @show norm(P' * X_cg) norm(X_cg - X_true)

    return hist
end