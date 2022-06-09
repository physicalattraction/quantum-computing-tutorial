from math import pi
from qiskit import QuantumCircuit, Aer, IBMQ

from qiskit_3_7_shor_algorithm import factorize
from utils import draw_quantum_circuit, print_histogram_from_real, load_transpiled_quantum_circuit

print('Qubit in state')
qc = QuantumCircuit(1, name='QC in initial state')
draw_quantum_circuit(qc, draw_bloch_sphere=True, draw_final_state=True, draw_unitary=True)

# print('Gates')
# qc = QuantumCircuit(1, name='Rotation around X')
# qc.x(0)
# draw_quantum_circuit(qc, draw_bloch_sphere=True)

# qc = QuantumCircuit(1, name='√X')
# qc.sx(0)
# draw_quantum_circuit(qc, draw_bloch_sphere=True, draw_final_state=True, draw_unitary=True)

# qc = QuantumCircuit(1, name='Hadamard')
# qc.h(0)
# draw_quantum_circuit(qc, draw_bloch_sphere=True, draw_final_state=True, draw_unitary=True)

# print('Measurements')
# qc = QuantumCircuit(1, name='QC in initial state')
# draw_quantum_circuit(qc, draw_circuit=False, draw_bloch_sphere=True, draw_histogram=True)
#
# qc = QuantumCircuit(1, name='Rotation around X')
# qc.x(0)
# draw_quantum_circuit(qc, draw_circuit=False, draw_bloch_sphere=True, draw_histogram=True)
#
# qc = QuantumCircuit(1, name='Hadamard')
# qc.h(0)
# draw_quantum_circuit(qc, draw_circuit=False, draw_bloch_sphere=True, draw_histogram=True)
#
# qc = QuantumCircuit(1, name='Rotation 1/4')
# qc.rx(pi / 4, 0)
# draw_quantum_circuit(qc, draw_circuit=False, draw_bloch_sphere=True, draw_histogram=True)
#
# qc = QuantumCircuit(1, name='Rotation 1/3')
# qc.rx(pi / 3, 0)
# draw_quantum_circuit(qc, draw_circuit=False, draw_bloch_sphere=True, draw_histogram=True)

# print('Multiple independent qubits')
# qc = QuantumCircuit(2, name='Independent qubits')
# qc.rx(pi / 4, 0)
# qc.rx(pi / 3, 1)
# draw_quantum_circuit(qc, draw_circuit=False, draw_bloch_sphere=True, draw_histogram=True)

# print('2-qubit gates')
# qc = QuantumCircuit(2, name='CNOT')
# qc.x(0)
# qc.cx(0, 1)  # Rotation around x of bit 1 if bit 0 == 1
# draw_quantum_circuit(qc, draw_circuit=False, draw_bloch_sphere=True, draw_histogram=True)
#
# print('Entangled qubits')
# qc = QuantumCircuit(2, name='Entangled CNOT')
# qc.h(0)
# qc.cx(0, 1)  # Rotation around x of bit 1 if bit 0 == 1
# draw_quantum_circuit(qc, draw_circuit=False, draw_bloch_sphere=True, draw_histogram=True)

# print('Building up gates')
# qc = QuantumCircuit(2, name='Controlled NOT with 2 Hadamards')
# qc.h(1)
# qc.cx(0, 1)
# qc.h(1)
# draw_quantum_circuit(qc, draw_unitary=True)
#
# qc = QuantumCircuit(2, name='Controlled Z')
# qc.cz(0, 1)
# draw_quantum_circuit(qc, draw_unitary=True)

# print('Backend operations')
# backend = Aer.get_backend('aer_simulator')
# print(backend.name(), backend.configuration().basis_gates)

# print('Backends')
# IBMQ.load_account()
# provider = IBMQ.get_provider(hub='ibm-q')
# backends = provider.backends()
# backend_names = [be.name() for be in backends]
# print('Backends available: ', backend_names)
# for name in backend_names:
#     backend = next(be for be in backends if be.name() == name)
#     print(backend.name(), backend.configuration().basis_gates)

# print('Build a quantum circuit')
# qc = QuantumCircuit(4, 2, name='Erwins QC')
# qc.x(1)  # Initial state
# qc.barrier()
# qc.h(0)
# qc.h(1)
# qc.h(2)
# qc.x(3)
# qc.cx(2, 3)
# qc.cx(1, 3)
# qc.cx(0, 3)
# qc.barrier()
# qc.measure(2, 0)
# qc.measure(3, 1)
# draw_quantum_circuit(qc, draw_simulate=True, nr_shots=1024)
#
# print('Running on a real quantum computer')
# print_histogram_from_real(qc, nr_shots=1024)

# print('Look at transpiled quantum circuit')
# backend = IBMQ.get_provider(hub='ibm-q').get_backend('ibmq_lima')
# qc = load_transpiled_quantum_circuit(qc, backend)
# draw_quantum_circuit(qc)

# print("Shor's algorithm")
# print('Measured phase is s/r, with a**r mod N = 1, and s = randint(0, r-1)')
# factorize(15)
