import qiskit as q
from qiskit import Aer  # simulator framework for qiskit
from qiskit.tools import job_monitor
from qiskit.visualization import plot_histogram


def circuit_1():
    circuit = q.QuantumCircuit(2, 2)  # 2 qubits, 2 classical bits
    # Q 0, 0
    circuit.x(0)
    # Q 1, 0
    circuit.cx(0, 1)  # cnot, controlled not, flips second qubit value iff first qubit is a 1
    # Q 1, 1
    circuit.measure([0, 1], [0, 1])
    # C 1, 1
    return circuit


def circuit_2():
    circuit = q.QuantumCircuit(2, 2)  # 2 qubits, 2 classical bits
    circuit.h(0)
    circuit.cx(0, 1)  # cnot, controlled not, flips second qubit value iff first qubit is a 1
    circuit.measure([0, 1], [0, 1])
    print(circuit.draw())
    return circuit


# Draw the circuit as a PNG
# plot = circuit_1().draw(output='mpl')
# plot.show()

for backend in Aer.backends():
    print(backend)
sim_backend = Aer.get_backend('qasm_simulator')

circuit = circuit_2()
job = q.execute(circuit, backend=sim_backend, shots=500)
job_monitor(job)
result = job.result()
counts = result.get_counts(circuit)
plot_histogram([counts]).show()
