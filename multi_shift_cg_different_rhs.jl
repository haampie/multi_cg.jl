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

function rand_spd(m, λs = range(1.0, m, length=m))
    F = qr!(rand(m, m))
    A = (F.Q * Diagonal(λs)) * F.Q'
    return A
end

using IterativeSolvers

function example(m = 100, maxiters=10)
    A = rand_spd(m)
    x = ones(m)
    b = A * x

    @show size(A) size(x) size(b)

    x_1, hist = cg(A, b, log=true, maxiter=maxiters)

    x_2 = zeros(m)
    multi_shift_cg_reference!(x_2, A, b, maxiters=maxiters)

    return x_1, x_2
    @show norm(x_1 - x_2)
end