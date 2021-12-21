
from enum import Enum

import numpy as np
from cvxopt import matrix, solvers
from processing import vector_of_quants
import math


class KnowledgePatternManager:
    @staticmethod
    def checkConsistency(knowledgePattern):
        return KnowledgePatternManager.__getConsistencyChecker(knowledgePattern.type) \
            .isConsistent(knowledgePattern)

    @staticmethod
    def __getConsistencyChecker(type):
        if type == KnowledgePatternType.QUANTS:
            return QuantConsistencyChecker()
        elif type == KnowledgePatternType.DISJUNCTS:
            return DisjunctConsistencyChecker()
        elif type == KnowledgePatternType.CONJUNCTS:
            return ConjunctConsistencyChecker()
        else:
            raise TypeError("Correct type of knowledge pattern")

    @staticmethod
    def getProbabilityFormula(knowledgePattern, formulaPattern):
        size = knowledgePattern.size
        matrix = MatrixProducer.getConjunctsToQuantsMatrix(int(math.log(size, 2)))
        intervals = np.array(knowledgePattern.array, dtype=np.double)
        vector = FormulaManager.getQuantsVector(formulaPattern, int(math.log(size, 2)))
        return LinearProgrammingProblemSolver.findOptimalFormulaValues(matrix, intervals, size, vector)

    @staticmethod
    def __getEvidenceCorrector(type):
        if type == EvidencePatternType.DETERMINISTIC:
            return DeterministicEvidenceCorrector()
        elif type == EvidencePatternType.STOCHASTIC:
            return StochasticEvidenceCorrector()
        elif type == EvidencePatternType.INACCURATE:
            return InaccurateEvidenceCorrector()
    @staticmethod
    def correctEvidenceData(knowledgePattern, evidencePattern):
        return KnowledgePatternManager.__getEvidenceCorrector(evidencePattern.type).getCorrectData(knowledgePattern, evidencePattern)


class FormulaManager:
    @staticmethod
    def getQuantsVector(formulaPattern, size):
        return vector_of_quants(formulaPattern.string, size)
    @staticmethod
    def getFormulaForOptimise(knowledgePattern, evidencePattern):
        size = knowledgePattern.size
        size_evidence = 2**(evidencePattern.size)
        result_formula = np.zeros(size)
        vector = EvidenceManager.getSubIdealProbability(evidencePattern)
        I = MatrixProducer.getConjunctsToQuantsMatrix(evidencePattern.size)
        ideal = EvidenceManager.getSubIdeal(evidencePattern)
        for i in range(0, 2**evidencePattern.size):
            array = [[ideal[i]], [ideal[size_evidence - 1 - i]]]
            formula = MatrixProducer.getTMatrix(array, int(math.log(size, 2)))[0]
            formula = np.dot(formula, np.dot(I, vector)[i])
            result_formula += formula
        return result_formula
    @staticmethod
    def getConjunctstoQuantsVector(vector):
        return np.dot(MatrixProducer.getConjunctsToQuantsMatrix(int(math.log(len(vector), 2))), vector)
    @staticmethod
    def getFormulaForOptimiseIn(knowledgePattern, evidencePattern):
        size = knowledgePattern.size
        matrix = MatrixProducer.getConjunctsToQuantsMatrix(int(math.log(size, 2)))
        intervals = np.array(knowledgePattern.array, dtype=np.double)
        matrix_for_opt = FormulaManager.getSubIdealtoIdealMatrix(evidencePattern, knowledgePattern)
        size_evidence = 2 ** (evidencePattern.size)

        result_formula_min = np.zeros(2 **evidencePattern.size)
        result_formula_max = np.zeros(2 **evidencePattern.size)
        I = MatrixProducer.getConjunctsToQuantsMatrix(evidencePattern.size)
        ideal = EvidenceManager.getSubIdeal(evidencePattern)
        for i in range(0, 2**evidencePattern.size):
            array = [[ideal[i]], [ideal[size_evidence - 1 - i]]]
            formula = MatrixProducer.getTMatrix(array, int(math.log(size, 2)))[0]
            prob = LinearProgrammingProblemSolver.findOptimalConjunctsFormulaValues(matrix, intervals, size, formula).array
            result_formula_min += I[i]*prob[0]
            result_formula_max += I[i]*prob[1]
        result = np.vstack([result_formula_min, result_formula_max])

        return result

    @staticmethod
    def getSubIdealtoIdealMatrix(evidencePattern, knowledgePattern):
        I = MatrixProducer.getConjunctsToQuantsMatrix(evidencePattern.size)
        ideal = EvidenceManager.getSubIdeal(evidencePattern)
        Matrix = np.zeros((2 ** evidencePattern.size, knowledgePattern.size), dtype = np.double)
        for i in range(0, 2 ** evidencePattern.size):
            for j in range(0, 2 **evidencePattern.size):
                Matrix[i][int(ideal[j])] = I[i][j]

        return Matrix















class EvidenceManager:
    @staticmethod
    def getConjunctsVector(evidencePattern):
        arr_conj = []
        num_conj = 0
        p_arr = evidencePattern.p_array
        for i in range(len(p_arr)):
            if p_arr[i] == 0: continue #?
            num_conj += pow(2, p_arr[i] - 1)
        arr_conj.append(num_conj)

        num_conj = 0
        m_arr = evidencePattern.m_array
        for i in range(len(m_arr)):
            num_conj += pow(2, p_arr[i] - 1)
        arr_conj.append(num_conj)
        return np.array(arr_conj)
    @staticmethod
    def getProbabilityOfDeterministicEvidence(knowledgePattern, mas):
        size = knowledgePattern.size
        matrix = MatrixProducer.getConjunctsToQuantsMatrix(int(math.log(size, 2)))
        intervals = np.array(knowledgePattern.array, dtype=np.double)
        vector = MatrixProducer.getTMatrix(mas, int(math.log(size, 2)))[0].tolist()

        return LinearProgrammingProblemSolver.findOptimalConjunctsFormulaValues(matrix, intervals, size, vector)

    @staticmethod
    def getProbabilityofStochasticEvidence(knowledgePattern, evidencePattern):
        size = knowledgePattern.size
        matrix = MatrixProducer.getConjunctsToQuantsMatrix(int(math.log(size, 2)))
        intervals = np.array(knowledgePattern.array, dtype=np.double)
        vector = FormulaManager.getFormulaForOptimise(knowledgePattern, evidencePattern)
        return LinearProgrammingProblemSolver.findOptimalConjunctsFormulaValues(matrix, intervals, size, vector)

    @staticmethod
    def getProbabilityofInaccurateEvidence(knowledgePattern, evidencePattern):
        size = evidencePattern.size
        matrix = MatrixProducer.getConjunctsToQuantsMatrix(evidencePattern.size)
        vectors = FormulaManager.getFormulaForOptimiseIn(knowledgePattern, evidencePattern)


        intervals = EvidenceManager.getSubIdealIntervalProbability(evidencePattern)

        return LinearProgrammingProblemSolver.findOptimalConjunctsFormulaValuesIn(matrix, intervals, size, vectors) 

    @staticmethod
    def getSubIdealProbability(evidencePattern):
        vector = np.ones(2 ** evidencePattern.size)

        array = evidencePattern.arr
        for i in range(0, 2**evidencePattern.size-1):
            vector[i+1] = array[i][1]
        return vector
    @staticmethod
    def getSubIdealIntervalProbability(evidencePattern):
        vector_min = np.ones(2 ** evidencePattern.size)
        vector_max = np.ones(2 ** evidencePattern.size)

        array = evidencePattern.arr
        for i in range(0, 2**evidencePattern.size-1):
            vector_min[i+1] = array[i][1]
            vector_max[i+1] = array[i][2]
        vector = []
        vector.append(vector_min)
        vector.append(vector_max)
        return vector



    @staticmethod
    def getSubIdeal(evidencePattern):
        vector = np.zeros(2 ** evidencePattern.size)
        array = evidencePattern.arr
        for i in range(0, 2**evidencePattern.size-1):
            vector[i+1] = array[i][0]
        return vector



class EvidencePatternType(Enum):
    DETERMINISTIC = 'deterministic',
    STOCHASTIC = 'stochastic',
    INACCURATE = 'inaccurate'

class KnowledgePatternType(Enum):
    QUANTS = 'quants',
    DISJUNCTS = 'disjuncts',
    CONJUNCTS = 'conjuncts'


class ConsistencyChecker:
    @staticmethod
    def isConsistent(knowledgePattern):
        raise NotImplementedError("It's a method of abstract class, use appropriate implementation")

class EvidenceCorrector:
    @staticmethod
    def getCorrextData(knowledgePattern, evidencePattern):
        raise NotImplementedError("It's a method of abstract class, use appropriate implementation")
class DeterministicEvidenceCorrector(EvidenceCorrector):
    @staticmethod
    def getCorrectData(knowledgePattern, evidencePattern):
        # разобраться с 1 и нулем
        size = knowledgePattern.size
        matrix = MatrixProducer.getConjunctsToQuantsMatrix(int(math.log(size, 2)))
        intervals = np.array(knowledgePattern.array, dtype=np.double)
        return LinearProgrammingProblemSolver.findOptimalEvidenceValues(matrix, intervals, size, MatrixProducer.getEvidencevector(evidencePattern.arr, int(math.log(size, 2))), intervals, MatrixProducer.getTMatrix(evidencePattern.arr, int(math.log(size, 2))))
class StochasticEvidenceCorrector(EvidenceCorrector):
    @staticmethod
    def getCorrectData(knowledgePattern, evidencePattern):
        size = knowledgePattern.size
        size_evidence = 2 ** (evidencePattern.size)
        result = [[0, 0] for i in range(knowledgePattern.size)]

        vector = EvidenceManager.getSubIdealProbability(evidencePattern) #p_ca
        I = MatrixProducer.getConjunctsToQuantsMatrix(int(math.log(knowledgePattern.size, 2)))
        I_1 = MatrixProducer.getConjunctsToQuantsMatrix(evidencePattern.size)
        vector_quants = np.dot(I_1, vector)
        ideal = EvidenceManager.getSubIdeal(evidencePattern)
        intervals = np.array(knowledgePattern.array, dtype=np.double)
        for i in range(0, 2 ** evidencePattern.size):
            array = [[ideal[i]], [ideal[size_evidence - 1 - i]]]
            divider = MatrixProducer.getTMatrix(array, int(math.log(size, 2)))[0]
            numerator = MatrixProducer.getTMatrix(array, int(math.log(size, 2)))

            ideal_ = LinearProgrammingProblemSolver.findOptimalStochasticEvidenceValues(I, intervals, size, numerator, divider)
            if len(ideal_) == 0:
                return EvidenceCorrectorResult(False, [])
            for j in range(size):
                result[j][0] += round(vector_quants[i] * ideal_[j][0], 3)
                result[j][1] += round(vector_quants[i] * ideal_[j][1], 3)

        if result[0][0] == 0: return EvidenceCorrectorResult(False, [])
        return EvidenceCorrectorResult(True, result)

class InaccurateEvidenceCorrector(EvidenceCorrector):
    @staticmethod
    def getCorrectData(knowledgePattern, evidencePattern):
        size = knowledgePattern.size
        size_evidence = 2 ** (evidencePattern.size)
        result_formula_min = np.zeros((size, size_evidence))
        result_formula_max = np.zeros((size, size_evidence))
        I = MatrixProducer.getConjunctsToQuantsMatrix(int(math.log(knowledgePattern.size, 2)))
        I_1 = MatrixProducer.getConjunctsToQuantsMatrix(evidencePattern.size)
        #vector_quants = np.dot(I_1, vector)
        ideal = EvidenceManager.getSubIdeal(evidencePattern)
        intervals = np.array(knowledgePattern.array, dtype=np.double)
        for i in range(0, 2 ** evidencePattern.size):
            array = [[ideal[i]], [ideal[size_evidence - 1 - i]]]
            divider = MatrixProducer.getTMatrix(array, int(math.log(size, 2)))[0]
            numerator = MatrixProducer.getTMatrix(array, int(math.log(size, 2)))
            ideal_ = LinearProgrammingProblemSolver.findOptimalStochasticEvidenceValues(I, intervals, size, numerator, divider)
            if len(ideal_) == 0:
                return EvidenceCorrectorResult(False, [])

            for j in range(size):
                result_formula_min[j] += I_1[i] * ideal_[j][0]
                result_formula_max[j] += I_1[i] * ideal_[j][1]


        return LinearProgrammingProblemSolver.findOptimalInaccurateEvidenceValues(I_1, EvidenceManager.getSubIdealIntervalProbability(evidencePattern),size, size_evidence, result_formula_min, result_formula_max)









class QuantConsistencyChecker(ConsistencyChecker):
    @staticmethod
    def isConsistent(knowledgePattern):
        size = knowledgePattern.size
        matrix = MatrixProducer.getIdentityMatrix(size)
        intervals = np.array(knowledgePattern.array, dtype=np.double)
        result = LinearProgrammingProblemSolver.findOptimalValues(matrix, intervals, size)
        if result.consistent:
            result = LinearProgrammingProblemSolver.findNormalizedOptimalValues(np.array(result.array, dtype=np.double),
                                                                                size)
        return result


class ConjunctConsistencyChecker(ConsistencyChecker):
    @staticmethod
    def isConsistent(knowledgePattern):
        size = knowledgePattern.size
        matrix = MatrixProducer.getConjunctsToQuantsMatrix(int(math.log(size, 2)))
        intervals = np.array(knowledgePattern.array, dtype=np.double)
        return LinearProgrammingProblemSolver.findOptimalValues(matrix, intervals, size)


class DisjunctConsistencyChecker(ConsistencyChecker):
    @staticmethod
    def isConsistent(knowledgePattern):
        size = knowledgePattern.size
        matrix = MatrixProducer.getDisjunctsToQuantsMatrix(int(math.log(size, 2)))
        intervals = np.array(knowledgePattern.array, dtype=np.double)
        return LinearProgrammingProblemSolver.findOptimalValues(matrix, intervals, size)


class MatrixProducer:
    @staticmethod
    def getDisjunctsToQuantsMatrix(n):
        return np.linalg.inv(MatrixProducer.getQuantsToDisjunctsMatrix(n))

    @staticmethod
    def getQuantsToDisjunctsMatrix(n):
        if n == 0:
            return np.array([1], dtype=np.double)
        elif n == 1:
            return np.array([[1, 1], [0, 1]], dtype=np.double)
        else:
            k = MatrixProducer.getQuantsToDisjunctsMatrix(n - 1)
            i = np.ones((2 ** (n - 1), 2 ** (n - 1)), dtype=np.double)
            k_o = k.copy()
            k_o[0] = [0] * 2 ** (n - 1)
            return np.block([[k, k], [k_o, i]])

    @staticmethod
    def getConjunctsToQuantsMatrix(n):
        if n == 0:
            return np.array([1], dtype=np.double)
        elif n == 1:
            return np.array([[1, -1], [0, 1]], dtype=np.double)
        else:
            i = MatrixProducer.getConjunctsToQuantsMatrix(n - 1)
            o = np.zeros((2 ** (n - 1), 2 ** (n - 1)), dtype=np.double)
            return np.block([[i, (-1) * i], [o, i]])

    @staticmethod
    def getIdentityMatrix(size):
        return np.eye(size, dtype=np.double)
    @staticmethod
    def getTMatrix(mas, size):
        matrix = np.array([1])
        I_1 = MatrixProducer.getConjunctsToQuantsMatrix(1)
        J_1 = np.linalg.inv(I_1)
        H_p = np.array([[0, 0], [0, 1]])
        H_m = np.array([[1, 0], [0, 0]])
        H = MatrixProducer.getIdentityMatrix(2)
        for i in range(size, 0, -1):
            if i == 0:
                matrix = np.kron(matrix, np.dot(np.dot(J_1, H_p), I_1))
                continue
            if i in mas[0]:
                matrix = np.kron(matrix, np.dot(np.dot(J_1, H_p), I_1))

            elif i in mas[1]:
                matrix = np.kron(matrix, np.dot(np.dot(J_1, H_m), I_1))
            else:
                matrix = np.kron(matrix, np.dot(np.dot(J_1, H), I_1))

        return matrix

    @staticmethod
    def getEvidencevector(mas, size):
        return MatrixProducer.getTMatrix(mas, size)[0]


class LinearProgrammingProblemSolver:

    @staticmethod
    def findOptimalFormulaValues(matrixs, array, size, vector):
        a = np.vstack(((-1) * matrixs, (-1) * np.eye(size, dtype=np.double), np.eye(size, dtype=np.double)))
        a = matrix(a)
        b = np.hstack((np.zeros(size, dtype=np.double), (-1) * array[:, 0], array[:, 1]))
        b = matrix(b)
        c = np.dot(np.array(MatrixProducer.getConjunctsToQuantsMatrix(int(math.log(len(vector), 2))).transpose()), vector)
        c = matrix(c)
        return LinearProgrammingProblemSolver.optimizeForFormula(a, b, c)

    @staticmethod
    def findOptimalConjunctsFormulaValues(matrixs, array, size, vector):
        a = np.vstack(((-1) * matrixs, (-1) * np.eye(size, dtype=np.double), np.eye(size, dtype=np.double)))
        a = matrix(a)
        b = np.hstack((np.zeros(size, dtype=np.double), (-1) * array[:, 0], array[:, 1]))
        b = matrix(b)
        c = np.array(vector)
        c = matrix(c)
        return LinearProgrammingProblemSolver.optimizeForFormula(a, b, c)


    @staticmethod
    def findOptimalValues(matrixs, array, size):
        a = np.vstack(((-1) * matrixs, (-1) * np.eye(size, dtype=np.double), np.eye(size, dtype=np.double)))
        a = matrix(a)
        b = np.hstack((np.zeros(size, dtype=np.double), (-1) * array[:, 0], array[:, 1]))
        b = matrix(b)
        c = np.array(np.zeros(size, dtype=np.double))
        c = matrix(c)
        return LinearProgrammingProblemSolver.optimizeForMatrices(a, b, c, size, array)



    @staticmethod
    def findNormalizedOptimalValues(array, size):
        a = np.vstack(((-1) * np.ones(size, dtype=np.double), np.ones(size, dtype=np.double),
                       (-1) * np.eye(size, dtype=np.double), np.eye(size, dtype=np.double)))
        a = matrix(a)
        b = np.hstack(
            ((-1) * np.ones(1, dtype=np.double), np.ones(1, dtype=np.double), (-1) * array[:, 0], array[:, 1]))
        b = matrix(b)
        c = np.array(np.zeros(size, dtype=np.double))
        c = matrix(c)
        return LinearProgrammingProblemSolver.optimizeForMatrices(a, b, c, size, array)

    @staticmethod
    def optimizeForMatrices(a, b, c, size, intervals):
        solvers.options['show_progress'] = False
        _intervals = intervals.copy()
        for i in range(size):
            c[i] = 1
            sol = solvers.lp(c, a, b)
            if sol['status'] != 'optimal':
                return ConsistencyResult(False, [])
            _intervals[i][0] = round(sol['x'][i], 3)

            c[i] = -1

            sol = solvers.lp(c, a, b)
            if sol['status'] != 'optimal':
                return ConsistencyResult(False, [])
            _intervals[i][1] = round(sol['x'][i], 3)
            c[i] = 0
        return ConsistencyResult(True, _intervals.tolist())

    @staticmethod
    def optimizeForFormula(a, b, c):
        answer = np.zeros(2)
        solvers.options['show_progress'] = False
        sol = solvers.lp(c, a, b)
        if sol['status'] != 'optimal':
            return ProbabilityFormulaResult(False, [])
        ans = 0
        for i in range(len(c)):
            ans += sol['x'][i]*c[i]

        answer[0] = round(ans, 3)

        c = -1 * c
        solvers.options['show_progress'] = False
        sol = solvers.lp(c, a, b)
        ans = 0
        if sol['status'] != 'optimal':
            return ProbabilityFormulaResult(False, [])
        for i in range(len(c)):
            ans += sol['x'][i]*c[i]
        answer[1] = round(-ans, 3)
        return ProbabilityFormulaResult(True, answer.tolist())

    @staticmethod
    def findOptimalEvidenceValues(matrixs, array, size, vector, intervals, T):
        a = np.vstack(((-1) * matrixs, (-1) * np.eye(size, dtype=np.double), np.eye(size, dtype=np.double), vector, (-1) * vector, np.zeros(size, dtype=np.double)))
        t = np.hstack((np.zeros(size, dtype=np.double), array[:, 0], (-1) * array[:, 1], np.zeros(2, dtype=np.double), np.array([-1])))
        a = np.column_stack((a, t))
        a = matrix(a)
        b = np.hstack((np.zeros(3 * size, dtype=np.double), np.array([1]), np.array([-1]), np.array([0])))
        b = matrix(b)
        return LinearProgrammingProblemSolver.optimizeForEvidenceMatrices(a, b, size, intervals, T)
    @staticmethod
    def findOptimalInaccurateEvidenceValues(matrixs, intervals, size, size_evidence, min_vect, max_vect):
        a = np.vstack(((-1) * matrixs, (-1) * np.eye(size_evidence, dtype=np.double), np.eye(size_evidence, dtype=np.double)))
        a = matrix(a)
        b = np.hstack((np.zeros(size_evidence, dtype=np.double), (-1) *intervals[0], intervals[1]))
        b = matrix(b)
        return LinearProgrammingProblemSolver.optimizeForInaccurateEvidenceMatrices(a, b, np.array(min_vect), np.array(max_vect), size, size_evidence)
    @staticmethod
    def optimizeForInaccurateEvidenceMatrices(a, b, c1, c2, size, size_evidence):
        solvers.options['show_progress'] = False
        _intervals = np.zeros((size, 2))

        for i in range(size):

            c = np.double(c1[i])
            c = matrix(c)
            sol = solvers.lp(c, a, b)
            if sol['status'] != 'optimal':
                return EvidenceCorrectorResult(False, [])

            for j in range(size_evidence):
                _intervals[i][0] += np.round(sol['x'][j] * c[j], 3)
            c = (-1)*np.double(c2[i])
            c = matrix(c)

            sol = solvers.lp(c, a, b)
            if sol['status'] != 'optimal':
                return []
            _intervals[i][1] = np.double(0)
            for j in range(size_evidence):
                _intervals[i][1] += np.round(sol['x'][j], 3) * np.round(c[j], 3)
            _intervals[i][1] = np.round((-1) * _intervals[i][1], 3)
        return EvidenceCorrectorResult(True, _intervals.tolist())



    @staticmethod
    def findOptimalStochasticEvidenceValues(matrixs, array, size, numerators, divider):
        a = np.vstack(((-1) * matrixs, (-1) * np.eye(size, dtype=np.double), np.eye(size, dtype=np.double), divider, (-1) * divider, np.zeros(size, dtype=np.double)))
        t = np.hstack((np.zeros(size, dtype=np.double), array[:, 0], (-1) * array[:, 1], np.zeros(2, dtype=np.double),np.array([-1])))
        a = np.column_stack((a, t))
        a = matrix(a)
        b = np.hstack((np.zeros(3 * size, dtype=np.double), np.array([1]), np.array([-1]), np.array([0])))
        b = matrix(b)
        return LinearProgrammingProblemSolver.optimizeForStochasticEvidenceMatrices(a, b, size, array, numerators)
    @staticmethod
    def findOptimalConjunctsFormulaValuesIn(matrixs, intervals, size, vector):
        size = 2 ** size

        a = np.vstack(((-1) * matrixs, (-1) * np.eye(size, dtype=np.double), np.eye(size, dtype=np.double)))
        a = matrix(a)
        b = np.hstack((np.zeros(size, dtype=np.double), (-1) * intervals[0], intervals[1]))
        b = matrix(b)
        c = np.array(vector)

        return LinearProgrammingProblemSolver.optimizeForFormulaIn(a, b, c)
    @staticmethod
    def optimizeForFormulaIn(a, b, c):
        answer = np.zeros(2)
        solvers.options['show_progress'] = False
        c1 = matrix(c[0])

        sol = solvers.lp(c1, a, b)
        if sol['status'] != 'optimal':
            return ProbabilityFormulaResult(False, [])
        ans = 0
        for i in range(len(c)):
            ans += sol['x'][i] * c1[i]

        answer[0] = np.round(ans, 3)

        c1 = matrix((-1) *c[1])
        solvers.options['show_progress'] = False
        sol = solvers.lp(c1, a, b)
        ans = 0
        if sol['status'] != 'optimal':
            return ProbabilityFormulaResult(False, [])
        for i in range(len(c)):
            ans += sol['x'][i] * c1[i]
        answer[1] = np.round((-1)*ans, 3)
        return ProbabilityFormulaResult(True, answer.tolist())

    @staticmethod
    def optimizeForStochasticEvidenceMatrices(a, b, size, intervals, numerators):
        solvers.options['show_progress'] = False
        _intervals = intervals.copy()

        for i in range(size):

            c = np.double(numerators[i])

            c = np.hstack((c, np.array([0])))
            c = matrix(c)
            sol = solvers.lp(c, a, b)
            if sol['status'] != 'optimal':
                return EvidenceCorrectorResult(False, [])
            _intervals[i][0] = np.double(0)

            for j in range(size):
                _intervals[i][0] += np.round(sol['x'][j], 3)*c[j]
            c = (-1) * c

            sol = solvers.lp(c, a, b)
            if sol['status'] != 'optimal':
                return []
            _intervals[i][1] = np.double(0)
            for j in range(size):
                _intervals[i][1] += np.round(sol['x'][j], 3)*c[j]
            _intervals[i][1] = np.round((-1)*_intervals[i][1],3)
        return _intervals.tolist()

    @staticmethod
    def optimizeForEvidenceMatrices(a, b, size, intervals, T):
        solvers.options['show_progress'] = False
        _intervals = intervals.copy()


        for i in range(size):
            c = np.double(T[i])

            c = np.hstack((c, np.array([0])))
            c = matrix(c)
            sol = solvers.lp(c, a, b)
            if sol['status'] != 'optimal':
                return EvidenceCorrectorResult(False, [])
            _intervals[i][0] = np.double(0)
            for j in range(size):
                _intervals[i][0] += round(sol['x'][j], 3)*c[j]

            c = (-1) * c

            sol = solvers.lp(c, a, b)
            if sol['status'] != 'optimal':
                return EvidenceCorrectorResult(False, [])
            _intervals[i][1] = np.double(0)
            for j in range(size):
                _intervals[i][1] += round(sol['x'][j], 3)*c[j]
            _intervals[i][1] = (-1)*_intervals[i][1]
        return EvidenceCorrectorResult(True, _intervals.tolist())

class EvidenceCorrectorResult:
    def __init__(self, existence, array):
        self._existence = existence
        self._arr = array

    @property
    def array(self):
        if self._existence:
            return self._arr
        else:
            raise AttributeError('There is no have array, because evidence is incorrect')

    @property
    def existence(self):
        return self._existence



class ConsistencyResult:
    def __init__(self, consistent, arr):
        self._consistent = consistent
        self._arr = arr

    @property
    def array(self):
        if self._consistent:
            return self._arr
        else:
            raise AttributeError('There is no have array, because knowledge pattern is Consistency')

    @property
    def consistent(self):
        return self._consistent

class ProbabilityFormulaResult:
    def __init__(self, existance, arr):
        self._existance = existance
        self._arr = arr

    @property
    def array(self):
        if self._existance:
            return self._arr
        else:
            raise AttributeError('There is no have array, because task have not solution')

    @property
    def existance(self):
        return self._existance



class KnowledgePatternItem:
    def __init__(self, array):
        self._arr = array

    @property
    def type(self):
        raise NotImplementedError("It's a method of abstract class, use appropriate implementation")

    def getElement(self, index):
        raise NotImplementedError("It's a method of abstract class, use appropriate implementation")

    @property
    def array(self):
        return NotImplementedError("It's a method of abstract class, use appropriate implementation")

    @property
    def size(self):
        return NotImplementedError("It's a method of abstract class, use appropriate implementation")

class EvidencePatternItem:
    def type(self):
        raise NotImplementedError("It's a method of abstract class, use appropriate implementation")

class DeterministicEvidencePatternItem(EvidencePatternItem):
    type = EvidencePatternType.DETERMINISTIC
    def __init__(self, array):
        self._array = array

    @property
    def p_array(self):
        return self._array[0]

    @property
    def m_array(self):
        return self._array[1]
    @property
    def arr(self):
        return self._array

class StochasticEvidencePatternItem(EvidencePatternItem):
    type = EvidencePatternType.STOCHASTIC

    def __init__(self, array):
        self._array = array

    @property
    def arr(self):
        return self._array

    @property
    def size(self):
        return int(math.log(len(self._array) + 1, 2))

class InaccurateEvidencePatternItem(EvidencePatternItem):
    type = EvidencePatternType.INACCURATE

    def __init__(self, array):
        self._array = array

    @property
    def arr(self):
        return self._array

    @property
    def size(self):
        return int(math.log(len(self._array) + 1, 2)) #то сколько символов

class FormulaPatternItem:
    def __init__(self, string):
        self._string = string

    @property
    def string(self):
        return self._string

    @property
    def size(self):
        return len(self._string)





class QuantKnowledgePatternItem(KnowledgePatternItem):
    _type = KnowledgePatternType.QUANTS

    @property
    def type(self):
        return self._type

    def getElement(self, index):
        return self._arr[index]

    @property
    def array(self):
        return self._arr

    @property
    def size(self):
        return len(self._arr)


class DisjunctKnowledgePatternItem(KnowledgePatternItem):
    _type = KnowledgePatternType.DISJUNCTS

    @property
    def type(self):
        return self._type

    def getElement(self, index):
        return self._arr[index]

    @property
    def array(self):
        return self._arr

    @property
    def size(self):
        return len(self._arr)


class ConjunctKnowledgePatternItem(KnowledgePatternItem):
    _type = KnowledgePatternType.CONJUNCTS

    @property
    def type(self):
        return self._type

    def getElement(self, index):
        return self._arr[index]

    @property
    def array(self):
        return self._arr

    @property
    def size(self):
        return len(self._arr)








