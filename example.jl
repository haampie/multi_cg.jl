using LinearAlgebra

import LinearAlgebra: mul!, ldiv!
import Base: size
using Base: OneTo

"""
This is the linear operator
H(X) = A * X - B * X * D + γ * Q * (Q' * (B * X)
where D = Diagonal(λs)
products are evaluated as follows:
C ← α * A * B + β * C
tmp = D * B
C ← -α * S * tmp + 1 * C
"""
struct MatrixOp{N,Tv,TM<:AbstractMatrix{Tv},Tλ} <: AbstractMatrix{Tv}
    A::TM
    B::TM
    Q::TM
    BQ::TM
    λs::Tλ
end

function MatrixOp{N}(A::TM, B, Q, λs) where {N,TM} 
    Tv = eltype(A)
    return MatrixOp{N,Tv,TM,typeof(λs)}(A, B, Q, B * Q, λs)
end

Base.size(A::MatrixOp) = size(A.A)

struct PrecOp{Tv,TM<:AbstractVector{Tv}} <: AbstractMatrix{Tv}
    A_diag::TM
    B_diag::TM
    λs::TM
end

Base.size(P::PrecOp) = (length(P.A_diag), length(P.A_diag))

function ldiv!(P::PrecOp, x::AbstractVecOrMat)
    for j in OneTo(size(x, 2)), i in OneTo(size(x, 1))
        # make sure not to divide by 0
        # x[i, j] /= max(1.0, )
        val = P.A_diag[i] - P.λs[j] * P.B_diag[i]
        x[i, j] /= sqrt(val^2 + 0.01)
    end
end

function the_mul!(Y::AbstractVecOrMat, H::MatrixOp{1}, X::AbstractVecOrMat, α::Number, β::Number)
    mul!(Y, H.A, X, α, β)
    mul!(Y, H.B * X, Diagonal(view(H.λs, axes(X, 2))), -α, 1.0)

    proj = H.BQ' * X

    for j in axes(proj, 2), i in axes(proj, 1)
        proj[i, j] *= H.λs[i] - H.λs[j]
    end

    mul!(Y, H.BQ, proj, α, 1.0)

    return Y
end

function the_mul!(Y::AbstractVecOrMat, H::MatrixOp{2}, X::AbstractVecOrMat, α::Number, β::Number)
    mul!(Y, H.A, X, α, β)
    tmp = X * Diagonal(view(H.λs, 1:size(X, 2)))
    mul!(Y, H.B, tmp, -α, 1)
    Y .+= (last(H.λs) + 1) .* (H.B * (H.Q * (H.Q' * (H.B * X))))
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

function block_cg!(X, A, P, B, U, C; maxiters = 10, tol=1e-8)
    m, n = size(X)

    @assert m ≥ n
    @assert size(X, 1) == size(B, 1) == size(U, 1) == size(C, 1)
    @assert size(X, 2) == size(B, 2) == size(U, 2) == size(C, 2)

    fill!(U, 0)

    # Use B effectively as the residual block-vector
    # B = B - A * X
    mul!(B, A, X, -1.0, 1.0)

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
        active = OneTo(num_unconverged)
        @views ldiv!(C[:, active], P, B[:, active])
        @views copyto!(ρs_old[active], ρs[active])

        for i = 1:num_unconverged
            ρs[i] = dot(view(C, :, i), view(B, :, i))
            push!(residual_history[ids[i]], sqrt(ρs[i]))
        end

        not_converged = [i for (i, v) in enumerate(view(ρs, OneTo(num_unconverged))) if v > tol^2]

        num_unconverged = length(not_converged)
        active = OneTo(num_unconverged)

        isempty(not_converged) && break

        # Move everything contiguously to the front.
        repack!(ids, not_converged)
        repack!(U, not_converged)
        repack!(C, not_converged)
        repack!(B, not_converged)
        repack!(A, not_converged)
        repack!(P, not_converged)
        repack!(ρs, not_converged)
        repack!(ρs_old, not_converged)

        # In the first iteration we have U == 0, so no need for an axpy.
        if iter == 1
            @views copyto!(U[:, active], C[:, active])
        else
            for i = 1:num_unconverged
                α = ρs[i] / ρs_old[i]
                @views axpby!(1.0, C[:, i], α, U[:, i])
            end
        end

        @views mul!(C[:, active], A, U[:, active])

        for i = 1:num_unconverged
            @views σs[i] = dot(U[:, i], C[:, i])
        end

        for i = 1:num_unconverged
            α = ρs[i] / σs[i]
            @views axpy!(α, U[:, i], X[:, ids[i]])
            @views axpy!(-α, C[:, i], B[:, i])
        end
    end

    return residual_history
end

function rand_spd(m, λs = range(1.0, m, length=m))
    F = qr!(rand(m, m))
    A = (F.Q * Diagonal(λs)) * F.Q'
    return A
end

using Plots

function standard_evp(m, n; maxiters = 50, reltol=1e-7)
    A, B = rand_spd(m), Matrix(1.0I, m, m)
    λs, Q = eigen(A)

    # 𝓟 = P * P' is a projector onto the first m eigenvectors.
    # 𝓟 * 𝓟 = P * (P' * P) * P' = P * P' = 𝓟
    # and so I - 𝓟 = I - P * P' projects on the rest of the spectrum.
    P = Q[:, 1:n]

    pl = plot(title="Standard eigenproblem perturbation", xlabel="iteration", ylabel=raw"$\left\Vert r \right\Vert$")

    for implementation = 1:2
        println("Implementation ", implementation)
        H = MatrixOp{implementation}(A, B, P, λs[1:n])

        # Create a true solution, and apply I - 𝓟 to remove a bunch of eigenvecs.
        X_true = rand(m, n)
        X_true .-= P * (P' * X_true)
        X_true .-= P * (P' * X_true) # apply it twice to avoid rounding errors

        rhs = H * X_true

        # The right-hand side should be orthogonal to P
        @show norm(P' * rhs)
        
        # Allocate a bunch of things to run block cg.
        prec = 1.0I # <- use an identity preconditioner
        # prec = PrecOp(diag(A), diag(B), λs[1:n])

        # initial guess
        X_cg = rand(m, n)
        X_cg .-= P * (P' * X_cg)
        X_cg .-= P * (P' * X_cg)

        # cg will destroy the B blockvector
        R = copy(rhs)

        # Auxilary stuff
        U, C = zeros(m, n), zeros(m, n)

        hist = block_cg!(X_cg, H, prec, R, U, C, maxiters=maxiters, tol=reltol)

        @show norm(P' * X_cg) norm(X_cg - X_true)

        plot!(pl, hist, color=(:red,:blue)[implementation], linestyle = (:dot, :solid)[implementation], linewidth=0.8, yscale=:log10, legend=false)
    end

    return pl
end

function generalized_evp(m, n; maxiters = 50)
    # Solve AX = BXΛ with X'BX = I
    A, B = rand_spd(m), rand_spd(m)
    λs, Q = eigen(Symmetric(A), Symmetric(B))

    # 𝓟 = P * P' * B is a projector onto the first m eigenvectors.
    # 𝓟 * 𝓟 = P * (P' * B * P) * P' * B = P * P' * B = 𝓟
    # and so I - 𝓟 = I - P * P' * B projects on the rest of the spectrum.
    P = Q[:, 1:n]

    pl = plot(title="Generalized eigenproblem perturbation", xlabel="iteration", ylabel=raw"$\left\Vert r \right\Vert$")

    for implementation = 1:2
        println("Implementation ", implementation)
        H = MatrixOp{implementation}(A, B, P, λs[1:n])

        # Create a true solution, and apply I - 𝓟 to remove a bunch of eigenvecs.
        X_true = rand(m, n)
        X_true .-= P * (P' * (B * X_true))
        X_true .-= P * (P' * (B * X_true)) # apply it twice to avoid rounding errors

        rhs = H * X_true

        # The right-hand side should be orthogonal to P
        @show norm(P' * rhs)

        # Create a zero initial guess for the correction
        X_cg = zeros(m, n)

        # Create a trivial preconditioner
        prec = 1.0I

        # Auxilary vectors
        U, C = zeros(m, n), zeros(m, n)

        # Run it
        hist = block_cg!(X_cg, H, prec, copy(rhs), U, C, maxiters=maxiters)

        @show norm(P' * X_cg) norm(X_cg - X_true)

        plot!(pl, hist, color=(:red,:blue)[implementation], linestyle = (:dot, :solid)[implementation], linewidth=0.8, yscale=:log10, legend=false)
    end

    return pl
end

function example_perturbation_eqn(m, n; maxiters = 50)
    # Solve AX = BXΛ with X'BX = I
    A, B = rand_spd(m), rand_spd(m)
    λs, X = eigen(Symmetric(A), Symmetric(B))

    # 𝓟 = Q * Q' * B is a projector onto the first m eigenvectors.
    # 𝓟 * 𝓟 = Q * (Q' * B * Q) * Q' * B = Q * Q' * B = 𝓟
    # and so I - 𝓟 = I - Q * Q' * B projects on the rest of the spectrum.
    Q = X[:, 1:n]

    # Create a perturbed matrix A.
    δA = rand_spd(m) .* 0.01

    # Create a right-hand side for the Newton linear system
    rhs = -δA * Q

    # Orthogonalize it (twice to be sure)
    rhs .-= B * (Q * (Q' * rhs))
    rhs .-= B * (Q * (Q' * rhs))

    # Create a zero initial guess for the correction
    X = zeros(m, n)

    # Create the matrix with projectors all around.
    H = MatrixOp{2}(A, B, Q, λs[1:n])

    # Create a trivial preconditioner
    prec = 1.0I

    # Auxilary vectors
    U, C = zeros(m, n), zeros(m, n)

    # Run it
    history = block_cg!(X, H, prec, copy(rhs), U, C, maxiters=maxiters)

    # Apply the correction
    Q′ = Q + X

    # Take a look at how well we did
    return A, B, Q′, λs[1:n]
end