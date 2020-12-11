using LinearAlgebra

function multi_shift_cg_reference!(x, A, b; maxiters=10, tol=1e-6)
    n = size(A, 1)

    V = zeros(n, maxiters + 1)
    H = zeros(maxiters + 1, maxiters)

    fill!(x, 0)

    u = zeros(n)

    copyto!(view(V, :, 1), b)
    β = norm(view(V, :, 1))
    view(V, :, 1) ./= β

    rhs = zeros(maxiters)
    rhs[1] = β

    LU = copy(H)

    for iter = 1:maxiters
        # new krylov basis vector
        mul!(view(V, :, iter + 1), A, view(V, :, iter))

        # orthogonalize it to previous vecs
        if iter > 1
            H[iter - 1, iter] = H[iter, iter - 1]
            view(V, :, iter + 1) .-= H[iter - 1, iter] .* view(V, :, iter - 1)
        end
        
        H[iter, iter] = dot(view(V, :, iter), view(V, :, iter + 1))
        view(V, :, iter + 1) .-= H[iter, iter] .* view(V, :, iter)

        # normalize
        H[iter + 1, iter] = norm(view(V, :, iter + 1))
        view(V, :, iter + 1) ./= H[iter+1, iter]
        
        # do a runnin LU-decomp, first copy new data over.
        if iter > 1
            LU[iter - 1, iter] = H[iter, iter - 1]
        end
        LU[iter, iter] = H[iter, iter]
        LU[iter + 1, iter] = H[iter + 1, iter]

        # do a step of LU decomp
        if iter > 1
            LU[iter, iter] -= LU[iter, iter - 1] * LU[iter - 1, iter]
        end

        LU[iter + 1, iter] /= LU[iter, iter]

        # update the right-hand side
        if iter > 1
            rhs[iter] -= LU[iter - 1, iter] * rhs[iter - 1]
        end
        rhs[iter] /= LU[iter, iter]

        # @show iter lu(H[1:iter,1:iter]).U' \ [β; zeros(iter-1)] - rhs[1:iter]

        if iter > 1
            # axpby
            u .= view(V, :, iter) .- LU[iter, iter - 1] .* u 
        else
            u .= view(V, :, iter)
        end

        x .+= rhs[iter] .* u
        
        @show norm(A * x - b)
    end

    return x, A, V, H, LU
end

function multi_shift_cg_efficient!(x, A, b; maxiters=10, tol=1e-6)
    n = size(A, 1)

    fill!(x, 0)

    u = zeros(n)
    r = zeros(n)
    t = zeros(n)
    v = zeros(n)

    copyto!(r, b)
    β = norm(r)
    r ./= β

    rhs = β

    h1 = 0.0
    
    LUᵢᵢ₋₁ = 0.0

    for iter = 1:maxiters
        # new krylov basis vector
        mul!(t, A, r)

        # orthogonalize it to previous vecs
        if iter > 1
            t .-= h1 .* v
        end
        
        h2 = dot(r, t)
        t .-= h2 .* r

        # normalize
        h3 = norm(t)
        t ./= h3
        
        # do a step of LU decomp
        LUᵢ₋₁ᵢ = h1
        LUᵢᵢ = h2
        if iter > 1
            LUᵢᵢ -= LUᵢᵢ₋₁ * h1
        end

        LUᵢ₊₁ᵢ = h3 / LUᵢᵢ

        # update the right-hand side
        if iter > 1
            rhs = -LUᵢ₋₁ᵢ * rhs
        end
        rhs /= LUᵢᵢ

        # @show iter lu(H[1:iter,1:iter]).U' \ [β; zeros(iter-1)] - rhs[1:iter]

        if iter > 1
            u .= r .- LUᵢᵢ₋₁ .* u 
        else
            u .= r
        end

        # axpy
        x .+= rhs .* u
        
        @show norm(A * x - b)

        # store stuff for the next iteration
        copyto!(v, r)
        copyto!(r, t)
        LUᵢᵢ₋₁ = LUᵢ₊₁ᵢ
        h1 = h3
    end

    return x
end

function multi_shift_cg_efficient_shift!(X, A, B, shifts; maxiters=10, tol=1e-6)
    n = size(A, 1)
    m = length(shifts)

    @assert size(X, 2) == size(B, 2) == m

    # Start out with a zero guess for now.
    fill!(X, 0)

    # new search directions, unique for every shift.
    u = zeros(n, m)

    # The Krylov subspace basis vectors, probably there's some optimization left
    # here to drop 1.
    r = zeros(n)
    t = zeros(n)
    v = zeros(n)

    # normalize the first krylov basis vec.
    copyto!(r, view(B, :, 1))
    β = norm(r)
    r ./= β

    # The running LU-decomposition, one lu-decomp per shift
    LUᵢᵢ₋₁ = zeros(m)
    h1 = 0.0
    rhs = zeros(m)

    residuals = [Float64[] for i = 1:m]

    for iter = 1:maxiters
        # new krylov basis vector
        mul!(t, A, r)

        # orthogonalize it to previous vecs
        if iter > 1
            t .-= h1 * v
        end
        
        h2 = dot(r, t)
        t .-= h2 .* r

        # normalize
        h3 = norm(t)
        t ./= h3
        
        # do a step of LU decomp
        LUᵢᵢ = h2 .+ shifts
        if iter > 1
            LUᵢᵢ .-= LUᵢᵢ₋₁ .* h1
        end

        # update the right-hand side
        for k = 1:m
            rhs[k] = (dot(r, view(B, :, k)) - h1 * rhs[k]) / LUᵢᵢ[k]
        end

        for k = 1:m
            if iter > 1
                u[:, k] .= r .- LUᵢᵢ₋₁[k] .* view(u, :, k) 
            else
                u[:, k] .= r
            end
        end

        LUᵢᵢ₋₁ .= h3 ./ LUᵢᵢ

        # axpy
        for k = 1:m
            X[:, k] .+= rhs[k] .* view(u, :, k)
        end
        
        for k = 1:m
            push!(residuals[k], norm(A * X[:, k] + shifts[k] .* X[:, k] .- B[:, k]))
        end

        # store stuff for the next iteration
        copyto!(v, r)
        copyto!(r, t)
        h1 = h3
    end

    return X, residuals
end

function rand_spd(m, λs = range(1.0, m, length=m))
    F = qr!(rand(m, m))
    A = (F.Q * Diagonal(λs)) * F.Q'
    return A
end

using IterativeSolvers

function example(;m = 100, shifts = [0.0, 1.0, 2.0], maxiters=10)
    n = length(shifts)

    A = rand_spd(m) + 10I
    B = rand(m, n)
    X = A \ B

    X_ref_cg = zeros(m, n)

    for (i, σ) in enumerate(shifts)
        X_ref_cg[:, i] .= cg(A + σ * I, B[:, i], maxiter=maxiters)
    end

    X_multi_cg = zeros(m, n)
    X_multi_cg, resnorms = multi_shift_cg_efficient_shift!(X_multi_cg, A, B, shifts, maxiters=maxiters)

    @show norm(X_ref_cg - X_multi_cg)

    return resnorms, A, B, X, X_ref_cg, X_multi_cg
end