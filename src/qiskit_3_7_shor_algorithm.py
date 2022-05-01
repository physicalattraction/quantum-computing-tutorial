"""
https://qiskit.org/textbook/ch-algorithms/shor.html
"""

from fractions import Fraction

import numpy as np
import pandas as pd
from math import pi, gcd
from qiskit import QuantumCircuit
from qiskit.circuit import ControlledGate

from utils import draw_quantum_circuit, print_histogram_from_real


def qft_dagger(n):
    """n-qubit QFTdagger the first n qubits in circ"""
    qc = QuantumCircuit(n)
    # Don't forget the Swaps!
    for qubit in range(n // 2):
        qc.swap(qubit, n - qubit - 1)
    for j in range(n):
        for m in range(j):
            qc.cp(-pi / float(2 ** (j - m)), m, j)
        qc.h(j)
    qc.name = "QFT†"
    return qc


def c_amod15(a, power) -> ControlledGate:
    """Controlled multiplication by a mod 15"""
    if a not in [2, 4, 7, 8, 11, 13]:
        raise ValueError("'a' must be 2, 4, 7, 8, 11 or 13")
    U = QuantumCircuit(4)
    for iteration in range(power):
        if a in [2, 13]:
            U.swap(0, 1)
            U.swap(1, 2)
            U.swap(2, 3)
        if a in [7, 8]:
            U.swap(2, 3)
            U.swap(1, 2)
            U.swap(0, 1)
        if a in [4, 11]:
            U.swap(1, 3)
            U.swap(0, 2)
        if a in [7, 11, 13]:
            for q in range(4):
                U.x(q)
    U = U.to_gate()
    U.name = f'{a}^{power} mod 15'
    c_U = U.control()
    return c_U


def qpe_amod15(a: int, use_real_device: bool = False):
    n_count = 8
    qc = QuantumCircuit(4 + n_count, n_count, name='qpe_amod')
    for q in range(n_count):
        qc.h(q)  # Initialize counting qubits in state |+>
    qc.x(3 + n_count)  # And auxiliary register in state |1>
    for q in range(n_count):  # Do controlled-U operations
        qc.append(c_amod15(a, 2 ** q),
                  [q] + [i + n_count for i in range(4)])
    qc.append(qft_dagger(n_count), range(n_count))  # Do inverse-QFT
    qc.measure(range(n_count), range(n_count))

    # Simulate Results
    if use_real_device:
        counts = print_histogram_from_real(qc)
    else:
        counts = draw_quantum_circuit(qc, draw_circuit=False, draw_simulate=True)
    measured_phases = []
    for output in counts:
        decimal = int(output, 2)  # Convert (base 2) string to decimal
        phase = decimal / (2 ** n_count)  # Find corresponding eigenvalue
        measured_phases.append(phase)

    phase = min([p for p in measured_phases if p != 0])
    print(f'Measured Phase for {a=}: {phase}')

    return phase


def factorize(nr: int, use_real_device: bool = False):
    if nr != 15:
        raise ValueError('This function has N=15 hardcoded in its circuit')
    guesses = {nr, 1}
    while len(guesses) <= 2:
        a = np.random.randint(2, 15)
        try:
            phase = qpe_amod15(a, use_real_device)  # Phase = s/r
        except ValueError:
            # Wrong value for a chosen. Try again
            return factorize(nr)
        frac = Fraction(phase).limit_denominator(15)  # Denominator should (hopefully!) tell us r
        s, r = frac.numerator, frac.denominator

        guesses.add(gcd(a ** (r // 2) - 1, nr))
        guesses.add(gcd(a ** (r // 2) + 1, nr))
        for guess in guesses:
            if guess not in {nr, 1}:
                if nr % guess == 0:
                    other_guess = nr // guess
                    print(f'My guess is: {guess} * {other_guess}')
                else:
                    print(f'This guess was wrong: {guess}')
        if guesses == {nr, 1}:
            print('No new guesses found')


if __name__ == '__main__':
    factorize(15)
