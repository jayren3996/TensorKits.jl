#---------------------------------------------------------------------------------------------------
# Kraus operator
#---------------------------------------------------------------------------------------------------
import LinearAlgebra: mul!
import Base: *, eltype
export kraus
struct Kraus{TL, TU}
    KL::Array{TL, 3}
    KU::Array{TU, 3}
    dir::Symbol
end
eltype(::Kraus{TL, TU}) where TL where TU = promote_type(TL, TU)
kraus(KL, KR; dir::Symbol=:r) = Kraus(Array(KL), Array(KR), dir)
kraus(KL; dir::Symbol=:r) = Kraus(Array(KL), conj.(KL), dir)
#---------------------------------------------------------------------------------------------------
function mul!(ρ::Matrix, k::Kraus, ρ0::AbstractMatrix)
    KL, KU, dir = k.KL, k.KU, k.dir
    if dir == :r
        @tensor ρ[:] = KL[-1, 3, 1] * ρ0[1, 2] * KU[-2, 3, 2]
    elseif dir == :l
        @tensor ρ[:] = KU[1, 3, -1] * ρ0[1, 2] * KL[2, 3, -2]
    else
        error("Illegal direction: $dir.")
    end
end

function mul!(ρ::Vector, k::Kraus, ρ0::AbstractVector)
    α = size(k.KL, 1)
    ρ_r = reshape(ρ, α, α)
    ρ0_r = reshape(ρ0, α, α)
    mul!(ρ_r, k, ρ0_r)
end
#---------------------------------------------------------------------------------------------------
function *(k::Kraus, ρ0::AbstractVecOrMat)
    ctype = promote_type(eltype.((k, ρ0))...)
    ρ = Array{ctype}(undef, size(ρ0))
    mul!(ρ, k, ρ0)
end

#---------------------------------------------------------------------------------------------------
# Eigen system using power iteration

# Find dominent eigensystem by iterative multiplication.
# Krylov method ensures Hermicity and semi-positivity.
#---------------------------------------------------------------------------------------------------
function complex_iter!(K, ρ1, ρ2)
    mul!(ρ1, K, ρ2)
    mul!(ρ2, K, ρ1)
    normalize!(ρ2)
end
#---------------------------------------------------------------------------------------------------
function real_iter!(K, ρ1, ρ2)
    mul!(ρ1, K, ρ2)
    mul!(ρ2, K, ρ1)
    ρ2 .+= ρ1
    normalize!(ρ2)
end
#---------------------------------------------------------------------------------------------------
function power_iteration(K, ρ1::Matrix, itr::Integer; method::Symbol=:r)
    ρ2 = similar(ρ1)
    if method == :r
        for i = 1:itr
            real_iter!(K, ρ1, ρ2)
        end
    elseif method == :c
        for i = 1:itr
            complex_iter!(K, ρ1, ρ2)
        end
    else
        error("Illegal method: $method.")
    end
    ρ2
end

#---------------------------------------------------------------------------------------------------
# steady state from identity mat
function steady_mat(
    K::Array{<:Number, 3}, 
    itr::Integer;
    dir::Symbol=:r
)
    α = size(K, 3)
    Kc = conj(K)
    kraus = Kraus(K, Kc, dir)
    ρ = Array{eltype(K)}(I(α))
    power_iteration(kraus, ρ, itr) |> Hermitian
end
#---------------------------------------------------------------------------------------------------
# Random fixed-point matrix.
function rand_fixed_mat(
    K::Array{<:Number, 3},
    itr::Integer;
    dir::Symbol=:r
)
    α = size(K, 3)
    Kc = conj(K)
    kraus = Kraus(K, Kc, dir)
    ρ = rand(ComplexF64, α, α) |> Hermitian |> Array
    power_iteration(kraus, ρ, itr) |> Hermitian
end
