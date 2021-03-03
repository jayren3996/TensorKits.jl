using TensorKits
using TensorKits: spin

# Create random iMPS
const BOND = 50

const mat = begin
    SS = spin([(1,"xx"), (1,"yy"), (1,"zz")], D=3)
    hamiltonian = SS + 1/3 * SS^2
    exp(-hamiltonian * 0.1)
end

# Create AKLT Hamiltonian and iTEBD engine
a = rand(ComplexF64, 1, 3, 1)
b = rand(ComplexF64, 1, 3, 1)
l = [1.0]

# Setup TEBD
for i=1:1000
    global BOND, mat, a, b, l
    for i=1:1000
        a,b,l = itebd2!(mat, a, b, l, bound=BOND)
    end
end

# Exact AKLT ground state
aklt = begin
    aklt_tensor = zeros(2,3,2)
    aklt_tensor[1,1,2] = +sqrt(2/3)
    aklt_tensor[1,2,1] = -sqrt(1/3)
    aklt_tensor[2,2,2] = +sqrt(1/3)
    aklt_tensor[2,3,1] = -sqrt(2/3)
    aklt_tensor
end