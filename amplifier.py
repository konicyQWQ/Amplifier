from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import IntegerComparator

import math
import numpy as np
import pandas as pd
import time
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("-dataset", type=str, required=True)
parser.add_argument("-P", type=str, default='P')
parser.add_argument("-G", type=str, default='G')
parser.add_argument("-M", type=str, default='M')
parser.add_argument("-P_V", type=int, default=1)
parser.add_argument("-op", type=str, required=True)
parser.add_argument("-s0", type=int, required=True)
parser.add_argument("-lowest", type=int, required=True)
parser.add_argument("-MultiTest", type=int, default=20)
parser.add_argument("-min_data_prob", type=float, default=0.1)
parser.add_argument("-min_shots", type=int)
parser.add_argument("-sup_shots", type=float, default=1.1)
parser.add_argument("-need_fix", type=int, default=0)
parser.add_argument("-fast-simu", type=int, default=0)

args = parser.parse_args()
dataset = args.dataset
P = args.P
G = args.G
M = args.M
P_V = args.P_V
MultiTest = args.MultiTest
min_data_prob = args.min_data_prob
first_sample_times = args.s0
lowest = args.lowest
min_shots = args.min_shots

def find_ones(num):
  result = []
  bit = 0
  while num > 0:
    if num % 2 == 1:
      result.append(bit)
    num = num // 2
    bit += 1
  return result

def int_to_binary_list_with_padding(n, bit_count):
    binary_list = []
    while n > 0:
        binary_list.insert(0, n % 2)
        n = n // 2
    while len(binary_list) < bit_count:
        binary_list.insert(0, 0)
    binary_list.reverse()
    return binary_list

def calc_grover_times(ratio):
    if ratio > 0.25:
        return 0, 1
    sin_theta = math.sqrt(ratio)
    theta = math.asin(sin_theta)
    i = 0
    now_theta = theta
    min_cost = 0
    min_i = 0
    min_prob = 0
    while True:
        i = i + 1
        now_theta = now_theta + theta * 2
        if now_theta > np.pi / 2:
            break
        prob = math.sin(now_theta)**2
        cost = 1 / prob * i
        if min_i == 0 or min_cost > cost:
            min_cost = cost
            min_i = i
            min_prob = prob
    return min_i, min_prob

def calc_origin_probs(ratio, gtimes):
    k = np.arcsin(np.sqrt(ratio)) / (2 * gtimes + 1)
    return np.sin(k)**2

def load_dataset():
    return pd.read_csv(dataset)

def dataset_to_PGMnumpy(dt: pd.DataFrame):
    return dt[[P, G, M]].values

def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0

if args.op != '=' and args.op != '>=' and args.op != '<':
    print("not support operator ", args.op)
    exit(1)

if not is_power_of_two(len(load_dataset())):
    print("data size must be the power of 2, but given data size is ", len(load_dataset()))
    exit(1)

def quantum_sample(data: np.ndarray, g_array: np.ndarray, is_delete: bool, shots: int, grover_times: int):
    if args.fast_simu:
        amplify_data = []
        other_data = []
        for i in range(len(data)):
            flag = True
            if args.op == '=':
                flag = flag and data[i][0] == P_V
            elif args.op == '>=':
                flag = flag and data[i][0] >= P_V
            elif args.op == '<':
                flag = flag and data[i][0] < P_V
            
            if not is_delete:
                flag = flag and data[i][1] in g_array
            else:
                flag = flag and not (data[i][1] in g_array)
            
            if flag:
                amplify_data.append({P: data[i][0], G: int(data[i][1]), M: data[i][2], "uid": i})
            else:
                other_data.append({P: data[i][0], G: int(data[i][1]), M: data[i][2], "uid": i})
        
        prob = math.sin((2 * grover_times + 1) * math.asin(math.sqrt(len(amplify_data) / len(data))))**2
        amplify_shots = 0
        for _ in range(shots):
            if random.random() < prob:
                amplify_shots += 1

        data1 = pd.DataFrame(amplify_data).sample(amplify_shots, replace=True)
        data2 = pd.DataFrame(other_data).sample(shots - amplify_shots, replace=True)
        return pd.concat([data1, data2])

    data_count = len(data)
    max_P, max_G, max_M = np.max(data[:, 0]), np.max(data[:, 1]), np.max(data[:, 2])
    idx_qbit_count = math.ceil(math.log2(data_count))
    data_P_qubit_count = math.ceil(math.log2(max_P + 1))
    data_G_qubit_count = math.ceil(math.log2(max_G + 1))
    data_M_qubit_count = math.ceil(math.log2(max_M + 1))

    qubits_sum = idx_qbit_count + data_P_qubit_count + data_G_qubit_count + data_M_qubit_count + 3
    print("qubits usage: ", qubits_sum)
    if qubits_sum >= 25:
        print("qubits is too much, simulator will run very slow!")

    def qmem_simu(need_m: bool):
        idx_qb = QuantumRegister(idx_qbit_count, 'idx')
        p_qb = QuantumRegister(data_P_qubit_count, 'p')
        g_qb = QuantumRegister(data_G_qubit_count, 'g')
        m_qb = QuantumRegister(data_M_qubit_count, 'm')
        qc = QuantumCircuit(m_qb, g_qb, p_qb, idx_qb)
        for i in range(data_count):
            idx_qbit = int_to_binary_list_with_padding(i, idx_qbit_count)
            for j in range(idx_qbit_count):
                if idx_qbit[j] == 0:
                    qc.x(idx_qb[j])
            p_qbit = find_ones(data[i, 0])
            for j in p_qbit:
                qc.mcx(idx_qb, p_qb[j])
            g_qbit = find_ones(data[i, 1])
            for j in g_qbit:
                qc.mcx(idx_qb, g_qb[j])
            if need_m:
                m_qbit = find_ones(data[i, 2])  
                for j in m_qbit:
                    qc.mcx(idx_qb, m_qb[j])
            for j in range(idx_qbit_count):
                if idx_qbit[j] == 0:
                    qc.x(idx_qb[j])
        return qc

    cr = ClassicalRegister(data_P_qubit_count + data_G_qubit_count + data_M_qubit_count + idx_qbit_count, 'idx_cr')
    idx_qb = QuantumRegister(idx_qbit_count, 'idx')
    p_qb = QuantumRegister(data_P_qubit_count, 'p')
    g_qb = QuantumRegister(data_G_qubit_count, 'g')
    m_qb = QuantumRegister(data_M_qubit_count, 'm')
    p_anc_qb = QuantumRegister(1, 'p_anc')
    g_anc_qb = QuantumRegister(1, 'g_anc')
    anc_qb = QuantumRegister(1, 'anc')
    ic_sup_qb = QuantumRegister(data_P_qubit_count-1, 'ic_sup')

    qram = qmem_simu(need_m=False)
    oracle = QuantumCircuit(ic_sup_qb, anc_qb, g_anc_qb, p_anc_qb, m_qb, g_qb, p_qb, idx_qb)
    oracle.compose(qram, qubits=m_qb[:]+g_qb[:]+p_qb[:]+idx_qb[:], inplace=True, wrap=True)
    oracle.x(anc_qb)
    oracle.h(anc_qb)
    # 处理 G
    for i in g_array:
        g_bit = int_to_binary_list_with_padding(i, data_G_qubit_count)
        for i in range(data_G_qubit_count):
            if g_bit[i] == 0:
                oracle.x(g_qb[i])
        oracle.mcx(g_qb[:], g_anc_qb)
        for i in range(data_G_qubit_count):
            if g_bit[i] == 0:
                oracle.x(g_qb[i])
    # 处理 P_V
    if args.op == '=':
        p_bit = int_to_binary_list_with_padding(P_V, data_P_qubit_count)
        for i in range(data_P_qubit_count):
            if p_bit[i] == 0:
                oracle.x(p_qb[i])
        oracle.mcx(p_qb[:], p_anc_qb)
        for i in range(data_P_qubit_count):
            if p_bit[i] == 0:
                oracle.x(p_qb[i])
    elif args.op == '>=':
        ic = IntegerComparator(num_state_qubits=data_P_qubit_count, value=P_V, geq=True)
        oracle.compose(ic, qubits=p_qb[:]+p_anc_qb[:]+ic_sup_qb[:], inplace=True)
    elif args.op == '<':
        ic = IntegerComparator(num_state_qubits=data_P_qubit_count, value=P_V, geq=False)
        oracle.compose(ic, qubits=p_qb[:]+p_anc_qb[:]+ic_sup_qb[:], inplace=True)
    else:
        print("not support operator ", args.op)
        exit(1)
    # 处理整体 anc_qb
    if is_delete:
        oracle.x(g_anc_qb)
    oracle.mcx(p_anc_qb[:]+g_anc_qb[:], anc_qb)
    if is_delete:
        oracle.x(g_anc_qb)
    # 还原 P_V
    if args.op == '=':
        p_bit = int_to_binary_list_with_padding(P_V, data_P_qubit_count)
        for i in range(data_P_qubit_count):
            if p_bit[i] == 0:
                oracle.x(p_qb[i])
        oracle.mcx(p_qb[:], p_anc_qb)
        for i in range(data_P_qubit_count):
            if p_bit[i] == 0:
                oracle.x(p_qb[i])
    elif args.op == '>=':
        ic = IntegerComparator(num_state_qubits=data_P_qubit_count, value=P_V, geq=True)
        oracle.compose(ic.inverse(), qubits=p_qb[:]+p_anc_qb[:]+ic_sup_qb[:], inplace=True)
    elif args.op == '<':
        ic = IntegerComparator(num_state_qubits=data_P_qubit_count, value=P_V, geq=False)
        oracle.compose(ic.inverse(), qubits=p_qb[:]+p_anc_qb[:]+ic_sup_qb[:], inplace=True)
    # 还原 g_anc_qb
    for i in g_array:
        g_bit = int_to_binary_list_with_padding(i, data_G_qubit_count)
        for i in range(data_G_qubit_count):
            if g_bit[i] == 0:
                oracle.x(g_qb[i])
        oracle.mcx(g_qb[:], g_anc_qb)
        for i in range(data_G_qubit_count):
            if g_bit[i] == 0:
                oracle.x(g_qb[i])
    oracle.h(anc_qb)
    oracle.x(anc_qb)
    oracle.compose(qram, qubits=m_qb[:]+g_qb[:]+p_qb[:]+idx_qb[:], inplace=True, wrap=True)

    grover = QuantumCircuit(ic_sup_qb, anc_qb, g_anc_qb, p_anc_qb, m_qb, g_qb, p_qb, idx_qb)
    grover.compose(oracle, inplace=True)
    grover.h(idx_qb)
    grover.x(idx_qb)
    grover.h(idx_qb[0])
    grover.mcx(idx_qb[1:], idx_qb[0])
    grover.h(idx_qb[0])
    grover.x(idx_qb)
    grover.h(idx_qb)

    qc = QuantumCircuit(ic_sup_qb, anc_qb, g_anc_qb, p_anc_qb, m_qb, g_qb, p_qb, idx_qb, cr)
    qc.h(idx_qb)
    for i in range(grover_times):
        qc.compose(grover, qubits=ic_sup_qb[:]+anc_qb[:]+g_anc_qb[:]+p_anc_qb[:]+m_qb[:]+g_qb[:]+p_qb[:]+idx_qb[:], inplace=True)
    qc.compose(qmem_simu(need_m=True), qubits=m_qb[:]+g_qb[:]+p_qb[:]+idx_qb[:], inplace=True, wrap=True)
    qc.measure(idx_qb[:]+m_qb[:]+g_qb[:]+p_qb[:], cr)

    def count_qc(qc: QuantumCircuit):
        backend = AerSimulator(method="statevector")
        t_qc = transpile(qc, backend)
        start_time = time.time()
        counts = backend.run(t_qc, shots=shots).result().get_counts()
        end_time = time.time()
        print(f"Quantum Circuit Runtime is {end_time - start_time}s")
        return counts

    counts = count_qc(qc)
    new_sample_data = []
    for j in counts:
        for _ in range(counts[j]):
            new_sample_data.append({
                P: int(j[:data_P_qubit_count], 2),
                G: int(j[data_P_qubit_count:data_P_qubit_count+data_G_qubit_count], 2),
                M: int(j[data_P_qubit_count+data_G_qubit_count:data_P_qubit_count+data_G_qubit_count+data_M_qubit_count] or '0', 2),
                "uid": int(j[data_P_qubit_count+data_G_qubit_count+data_M_qubit_count:], 2)
            })
    return pd.DataFrame(new_sample_data)

dt = load_dataset()
all_sample_times = [first_sample_times]
all_sample_amplify = [0]
all_group_prob = {}
all_delete_g = [[]]

# Step 1. 普通采样
sample_dt = dt.sample(first_sample_times, replace=False)
if args.op == '=':
    sample_dt_g_counts = (sample_dt[sample_dt[P] == P_V] if P_V != None else sample_dt)[G].value_counts()
elif args.op == '>=':
    sample_dt_g_counts = (sample_dt[sample_dt[P] >= P_V] if P_V != None else sample_dt)[G].value_counts()
elif args.op == '<':
    sample_dt_g_counts = (sample_dt[sample_dt[P] < P_V] if P_V != None else sample_dt)[G].value_counts()
sample_dt_g_prob = sample_dt_g_counts / len(sample_dt)
print(f"G   : {sample_dt_g_prob.index.values}\nprob: {sample_dt_g_prob.values}")
for i in range(len(sample_dt_g_prob)):
    all_group_prob[sample_dt_g_prob.index.values[i]] = sample_dt_g_prob.values[i]

multi_sample_dt = sample_dt.copy()

# Step 2.1. 寻找初次分割点
first_cut_point = 0
for i in range(0, sample_dt_g_prob.size):
    b = sample_dt_g_prob.iat[i]
    if b < 0.2:
        first_cut_point = i - 1
        break

if first_cut_point >= 0:
    print(f"经典采样范围: 0~{first_cut_point}, prob = {sample_dt_g_prob.iat[first_cut_point]}")

    # Step 2.2 完成经典采样并且更新概率和采样次数数组
    shots = int((lowest - sample_dt_g_counts.iat[first_cut_point]) / sample_dt_g_prob.iat[first_cut_point])
    if shots > 0:
        sample_dt = pd.concat([sample_dt, dt.sample(shots, replace=False)])
        if args.op == '=':
            sample_dt_g_counts = (sample_dt[sample_dt[P] == P_V] if P_V != None else sample_dt)[G].value_counts()
        elif args.op == '>=':
            sample_dt_g_counts = (sample_dt[sample_dt[P] >= P_V] if P_V != None else sample_dt)[G].value_counts()
        elif args.op == '<':
            sample_dt_g_counts = (sample_dt[sample_dt[P] < P_V] if P_V != None else sample_dt)[G].value_counts()
        sample_dt_g_prob = sample_dt_g_counts / len(sample_dt)
        for i in range(len(sample_dt_g_prob)):
            all_group_prob[sample_dt_g_prob.index.values[i]] = sample_dt_g_prob.values[i]
        print(f"经典采样次数: {shots}")
        all_sample_times.append(shots)
        all_sample_amplify.append(0)

def normal_recur_quantum_sample(g_probs: pd.Series):
    global sample_dt
    global multi_sample_dt
    global sample_dt_g_counts
    global all_sample_times
    global all_sample_amplify
    global all_group_prob
    global all_delete_g
    # Step 1. 寻找最佳分割点
    prob_sum = g_probs.sum()
    a = g_probs.iat[0]
    c = g_probs.iat[g_probs.size - 1]
    grover_times_1, grover_prob_1 = calc_grover_times(prob_sum)
    min_cost = 0
    min_i = -1
    for i in range(g_probs.size):
        b = g_probs.iat[i]
        x = g_probs[0:i+1].sum()
        y = prob_sum - x
        cost_1 = lowest / (b / prob_sum) / grover_prob_1 * grover_times_1
        if i == g_probs.size - 1:
            cost_2 = 0
        else:
            grover_times_2, grover_prob_2 = calc_grover_times(y)
            cost_2 = (lowest - lowest / (b / prob_sum) * (c / prob_sum)) / (c / y) / grover_prob_2 * grover_times_2
        cost = cost_1 + cost_2 * 0.8
        if min_i == -1 or min_cost >= cost:
            min_cost = cost
            min_i = i
    # Step 2. 量子采样
    shots = int((lowest - sample_dt_g_counts[g_probs.index[min_i]]) / (g_probs.iat[min_i] / prob_sum) / grover_prob_1)
    if min_shots and shots < min_shots:
        shots = min_shots
    shots = int(shots * args.sup_shots)
    print(f"本次 G 分类: {g_probs.index.values},\n概率: {g_probs.values}\n总和: {prob_sum}")
    print(f"本次 划分点: {g_probs.index[0:min_i+1].values}\n对整体进行 {shots} 次采样, 放大次数是: {grover_times_1}")
    all_sample_times.append(shots)
    all_sample_amplify.append(grover_times_1)
    shots = shots * MultiTest
    new_sample_data = quantum_sample(data=dataset_to_PGMnumpy(dt),
                   g_array=g_probs.index,
                   is_delete=False,
                   shots=shots,
                   grover_times=grover_times_1)
    multi_sample_dt = pd.concat([multi_sample_dt, new_sample_data])
    new_sample_data = new_sample_data.sample(int(shots / MultiTest), replace=False)
    # Step 3. 合并数据
    sample_dt = pd.concat([sample_dt, new_sample_data])
    if args.op == '=':
        sample_dt_g_counts = (sample_dt[sample_dt[P] == P_V] if P_V != None else sample_dt)[G].value_counts()
    elif args.op == '>=':
        sample_dt_g_counts = (sample_dt[sample_dt[P] >= P_V] if P_V != None else sample_dt)[G].value_counts()
    elif args.op == '<':
        sample_dt_g_counts = (sample_dt[sample_dt[P] < P_V] if P_V != None else sample_dt)[G].value_counts()

    # Step 4. 去掉不在 z_probs 里面的, 重新计算新的概率，留下 new_z_probs 里面的不满足 lowest 的
    new_sample_data_with_noreplace = new_sample_data.copy()
    new_sample_data_with_noreplace = new_sample_data_with_noreplace.drop_duplicates(subset='uid')
    if args.op == '=':
        new_sample_data_with_noreplace = new_sample_data_with_noreplace[new_sample_data_with_noreplace[P] == P_V] if P_V != None else new_sample_data_with_noreplace
    elif args.op == '>=':
        new_sample_data_with_noreplace = new_sample_data_with_noreplace[new_sample_data_with_noreplace[P] >= P_V] if P_V != None else new_sample_data_with_noreplace
    elif args.op == '<':
        new_sample_data_with_noreplace = new_sample_data_with_noreplace[new_sample_data_with_noreplace[P] < P_V] if P_V != None else new_sample_data_with_noreplace
    new_sample_data_with_noreplace = new_sample_data_with_noreplace[new_sample_data_with_noreplace[G].isin(g_probs.index)]
    # 更新概率数组
    if args.op == '=':
        new_sample_data = new_sample_data[new_sample_data[P] == P_V] if P_V != None else new_sample_data
    elif args.op == '>=':
        new_sample_data = new_sample_data[new_sample_data[P] >= P_V] if P_V != None else new_sample_data
    elif args.op == '<':
        new_sample_data = new_sample_data[new_sample_data[P] < P_V] if P_V != None else new_sample_data
    new_sample_data = new_sample_data[new_sample_data[G].isin(g_probs.index)]
    quantum_prob_sum = calc_origin_probs(len(new_sample_data) / int(shots / MultiTest), grover_times_1)
    if int(shots / MultiTest) > 30 and np.abs(quantum_prob_sum - prob_sum) / quantum_prob_sum > 0.04:
        prob_sum = quantum_prob_sum
    
    if int(shots / MultiTest) > 20 and int(shots / MultiTest) > new_sample_data_with_noreplace[G].value_counts().sum() * np.log2(new_sample_data_with_noreplace[G].value_counts().sum()):
        prob_sum = new_sample_data_with_noreplace[G].value_counts().sum() / len(dt)
    new_g_probs = new_sample_data_with_noreplace[G].value_counts() / len(new_sample_data_with_noreplace) * prob_sum
    for i in range(len(new_g_probs)):
        all_group_prob[new_g_probs.index.values[i]] = new_g_probs.values[i]
    
    new_g_probs = new_g_probs[~new_g_probs.index.isin(sample_dt_g_counts[sample_dt_g_counts > lowest * 0.8].index)]
    print(f"剩下需要递归的 G 分类: {new_g_probs.index.values}\n概率: {new_g_probs.values}")

    multi_sample_dt.to_csv('multi_sample_dt.csv')
    sample_dt.to_csv('sample_dt.csv')

    # Step 5. 递归处理(如果没有了则结束)
    if new_g_probs.size != 0:
        normal_recur_quantum_sample(new_g_probs)

def last_recur_quantum_sample(delete_g_array: np.ndarray, g_probs: pd.Series):
    global sample_dt
    global multi_sample_dt
    global sample_dt_g_counts
    global all_sample_times
    global all_sample_amplify
    global all_delete_g
    # Step 1. 寻找最佳分割点
    prob_sum = g_probs.sum()
    a = g_probs.iat[0]
    c = g_probs.iat[g_probs.size - 1]
    grover_times_1, grover_prob_1 = calc_grover_times(prob_sum)
    min_cost = 0
    min_i = -1
    for i in range(g_probs.size):
        b = g_probs.iat[i]
        x = g_probs[0:i+1].sum()
        y = prob_sum - x
        cost_1 = lowest / (b / prob_sum) / grover_prob_1 * grover_times_1
        if i == g_probs.size - 1:
            cost_2 = 0
        else:
            grover_times_2, grover_prob_2 = calc_grover_times(y)
            cost_2 = (lowest - lowest / (b / prob_sum) * (c / prob_sum)) / (c / y) / grover_prob_2 * grover_times_2
        cost = cost_1 + cost_2 * 0.9
        if min_i == -1 or min_cost >= cost:
            min_cost = cost
            min_i = i
    # Step 2. 量子采样
    shots = int((lowest - sample_dt_g_counts[g_probs.index[min_i]]) / (g_probs.iat[min_i] / prob_sum) / grover_prob_1)
    if min_shots and shots < min_shots:
        shots = min_shots
    shots = int(shots * args.sup_shots)
    print(f"本次 G 分类: {g_probs.index.values},\n概率: {g_probs.values}\n总和: {prob_sum}")
    print(f"本次 划分点: {g_probs.index[0:min_i+1].values}")
    print(f"排除前面的 G {delete_g_array} 后对整体进行 {shots} 次采样, 放大次数是: {grover_times_1}")
    all_sample_times.append(shots)
    all_sample_amplify.append(grover_times_1)
    shots = shots * MultiTest
    new_sample_data = quantum_sample(data=dataset_to_PGMnumpy(dt),
                   g_array=delete_g_array,
                   is_delete=True,
                   shots=shots,
                   grover_times=grover_times_1)
    all_delete_g.append(delete_g_array)
    multi_sample_dt = pd.concat([multi_sample_dt, new_sample_data])
    new_sample_data = new_sample_data.sample(int(shots / MultiTest), replace=False)
    # Step 3. 合并数据
    sample_dt = pd.concat([sample_dt, new_sample_data])
    if args.op == '=':
        sample_dt_g_counts = (sample_dt[sample_dt[P] == P_V] if P_V != None else sample_dt)[G].value_counts()
    elif args.op == '>=':
        sample_dt_g_counts = (sample_dt[sample_dt[P] >= P_V] if P_V != None else sample_dt)[G].value_counts()
    elif args.op == '<':
        sample_dt_g_counts = (sample_dt[sample_dt[P] < P_V] if P_V != None else sample_dt)[G].value_counts()

    # Step 4. 去掉不在 z_probs 里面的, 重新计算新的概率，留下 new_z_probs 里面的不满足 lowest 的
    new_sample_data_with_noreplace = new_sample_data.copy()
    new_sample_data_with_noreplace = new_sample_data_with_noreplace.drop_duplicates(subset='uid')
    if args.op == '=':
        new_sample_data_with_noreplace = new_sample_data_with_noreplace[new_sample_data_with_noreplace[P] == P_V] if P_V != None else new_sample_data_with_noreplace
    elif args.op == '>=':
        new_sample_data_with_noreplace = new_sample_data_with_noreplace[new_sample_data_with_noreplace[P] >= P_V] if P_V != None else new_sample_data_with_noreplace
    elif args.op == '<':
        new_sample_data_with_noreplace = new_sample_data_with_noreplace[new_sample_data_with_noreplace[P] < P_V] if P_V != None else new_sample_data_with_noreplace
    new_sample_data_with_noreplace = new_sample_data_with_noreplace[~new_sample_data_with_noreplace[G].isin(delete_g_array)]

    if args.op == '=':
        new_sample_data = new_sample_data[new_sample_data[P] == P_V] if P_V != None else new_sample_data
    elif args.op == '>=':
        new_sample_data = new_sample_data[new_sample_data[P] >= P_V] if P_V != None else new_sample_data
    elif args.op == '<':
        new_sample_data = new_sample_data[new_sample_data[P] < P_V] if P_V != None else new_sample_data
    new_sample_data = new_sample_data[~new_sample_data[G].isin(delete_g_array)]
    # 更新概率数组
    quantum_prob_sum = calc_origin_probs(len(new_sample_data) / int(shots / MultiTest), grover_times_1)
    if int(shots / MultiTest) > 70 and np.abs(quantum_prob_sum - prob_sum) / quantum_prob_sum > 0.02:
        prob_sum = quantum_prob_sum
    
    if int(shots / MultiTest) > new_sample_data_with_noreplace[G].value_counts().sum() * np.log2(new_sample_data_with_noreplace[G].value_counts().sum()):
        prob_sum = new_sample_data_with_noreplace[G].value_counts().sum() / len(dt)

    new_g_probs = new_sample_data_with_noreplace[G].value_counts() / len(new_sample_data_with_noreplace) * prob_sum
    for i in range(len(new_g_probs)):
        all_group_prob[new_g_probs.index.values[i]] = new_g_probs.values[i]
    
    new_delete_g_array = np.intersect1d(new_g_probs.index, sample_dt_g_counts[sample_dt_g_counts > lowest * 0.9].index)
    new_delete_g_array = np.union1d(new_delete_g_array, delete_g_array)
    new_g_probs = new_g_probs[~new_g_probs.index.isin(sample_dt_g_counts[sample_dt_g_counts > lowest * 0.9].index)]
    print(f"新的删除 G 数组: {new_delete_g_array}\n剩下需要递归的 G 分类: {new_g_probs.index.values}\n概率是 {new_g_probs.values}")

    multi_sample_dt.to_csv('multi_sample_dt.csv')
    sample_dt.to_csv('sample_dt.csv')

    # Step 5. 递归处理(如果没有了则检查剩下的是否达到阈值)
    if new_g_probs.size != 0:
        last_recur_quantum_sample(new_delete_g_array, new_g_probs)
    else:
        last_data_prob = 1 / (int(shots / MultiTest) * grover_prob_1) * prob_sum
        print(f"没有探测到的稀疏数据可能概率为 {last_data_prob}")
        if last_data_prob < min_data_prob:
            return
        else:
            # Step 6. 再次 Grover 检查
            while True:
                prob_sum = last_data_prob
                delete_g_array = new_delete_g_array
                shots = lowest
                if min_shots and shots < min_shots:
                    shots = min_shots
                shots = int(shots * args.sup_shots)
                grover_times_1, grover_probs_1 = calc_grover_times(prob_sum)
                print(f"排除前面的 G {delete_g_array} 后对整体进行 {shots} 次采样, 放大次数是: {grover_times_1}")
                all_sample_times.append(shots)
                all_sample_amplify.append(grover_times_1)
                shots = shots * MultiTest
                new_sample_data = quantum_sample(data=dataset_to_PGMnumpy(dt),
                            g_array=delete_g_array,
                            is_delete=True,
                            shots=shots,
                            grover_times=grover_times_1)
                all_delete_g.append(delete_g_array)
                multi_sample_dt = pd.concat([multi_sample_dt, new_sample_data])
                new_sample_data = new_sample_data.sample(int(shots / MultiTest), replace=False)
                # Step 3. 合并数据
                sample_dt = pd.concat([sample_dt, new_sample_data])
                if args.op == '=':
                    sample_dt_g_counts = (sample_dt[sample_dt[P] == P_V] if P_V != None else sample_dt)[G].value_counts()
                elif args.op == '>=':
                    sample_dt_g_counts = (sample_dt[sample_dt[P] >= P_V] if P_V != None else sample_dt)[G].value_counts()
                elif args.op == '<':
                    sample_dt_g_counts = (sample_dt[sample_dt[P] < P_V] if P_V != None else sample_dt)[G].value_counts()

                # Step 4. 去掉不在 z_probs 里面的, 重新计算新的概率，留下 new_z_probs 里面的不满足 lowest 的
                new_sample_data_with_noreplace = new_sample_data.copy()
                new_sample_data_with_noreplace = new_sample_data_with_noreplace.drop_duplicates(subset='uid')
                if args.op == '=':
                    new_sample_data_with_noreplace = new_sample_data_with_noreplace[new_sample_data_with_noreplace[P] == P_V] if P_V != None else new_sample_data_with_noreplace
                elif args.op == '>=':
                    new_sample_data_with_noreplace = new_sample_data_with_noreplace[new_sample_data_with_noreplace[P] >= P_V] if P_V != None else new_sample_data_with_noreplace
                elif args.op == '<':
                    new_sample_data_with_noreplace = new_sample_data_with_noreplace[new_sample_data_with_noreplace[P] < P_V] if P_V != None else new_sample_data_with_noreplace
                new_sample_data_with_noreplace = new_sample_data_with_noreplace[~new_sample_data_with_noreplace[G].isin(delete_g_array)]

                if args.op == '=':
                    new_sample_data = new_sample_data[new_sample_data[P] == P_V] if P_V != None else new_sample_data
                elif args.op == '>=':
                    new_sample_data = new_sample_data[new_sample_data[P] >= P_V] if P_V != None else new_sample_data
                elif args.op == '<':
                    new_sample_data = new_sample_data[new_sample_data[P] < P_V] if P_V != None else new_sample_data
                new_sample_data = new_sample_data[~new_sample_data[G].isin(delete_g_array)]
                # 更新概率数组
                quantum_prob_sum = calc_origin_probs(len(new_sample_data) / int(shots / MultiTest), grover_times_1)
                if int(shots / MultiTest) > 50 and np.abs(quantum_prob_sum - prob_sum) / quantum_prob_sum > 0.02:
                    prob_sum = quantum_prob_sum
                new_g_probs = new_sample_data_with_noreplace[G].value_counts() / len(new_sample_data_with_noreplace) * prob_sum
                for i in range(len(new_g_probs)):
                    all_group_prob[new_g_probs.index.values[i]] = new_g_probs.values[i]
                
                new_delete_g_array = np.intersect1d(new_g_probs.index, sample_dt_g_counts[sample_dt_g_counts > lowest * 0.9].index)
                new_delete_g_array = np.union1d(new_delete_g_array, delete_g_array)
                new_g_probs = new_g_probs[~new_g_probs.index.isin(sample_dt_g_counts[sample_dt_g_counts > lowest * 0.9].index)]
                print(f"新的删除 G 数组: {new_delete_g_array}\n剩下需要递归的 G 分类: {new_g_probs.index.values}\n概率是 {new_g_probs.values}")

                multi_sample_dt.to_csv('multi_sample_dt.csv')
                sample_dt.to_csv('sample_dt.csv')

                if new_g_probs.size == 0:
                    print(f"没有发现新分类")
                    last_data_prob = 1 / (int(shots / MultiTest) * grover_prob_1) * prob_sum
                    print(f"没有探测到的稀疏数据可能概率为 {last_data_prob}")
                    if last_data_prob < min_data_prob:
                        return
                    else:
                        continue
                else:
                    print(f"发现新 G 分类 {new_g_probs.index.values}, 概率是 {new_g_probs.values}")
                    last_recur_quantum_sample(new_delete_g_array, new_g_probs)
                break

# Step 2.3 排除已经采够的部分，按照20%划定寻找各部分，进行量子采样
sample_dt_g_prob_sum = sample_dt_g_prob.cumsum()
i = sample_dt_g_counts.index.get_loc((sample_dt_g_counts < lowest * 0.9).idxmax())
size = sample_dt_g_counts.size
for j in range(i, size):
    # Step 3.2 最后一部分，迭代量子采样，寻找稀疏数据
    if j == size - 1:
        last_recur_quantum_sample(sample_dt_g_prob[0:i].index, sample_dt_g_prob[i:j+1])
        break
    # Step 3.1 中间部分进行确定的量子采样([i, j]区间和要小于0.2)
    elif i == 0:
        if sample_dt_g_prob_sum.iat[j] <= 0.2 and sample_dt_g_prob_sum.iat[j + 1] > 0.2:
            normal_recur_quantum_sample(sample_dt_g_prob[i:j+1])
            i = j + 1
    elif sample_dt_g_prob_sum.iat[j] -  sample_dt_g_prob_sum.iat[i - 1] <= 0.2 and sample_dt_g_prob_sum.iat[j + 1] -  sample_dt_g_prob_sum.iat[i - 1] > 0.2:
        normal_recur_quantum_sample(sample_dt_g_prob[i:j+1])
        i = j + 1

# %%
multi_sample_dt.to_csv('multi_sample_dt.csv')
sample_dt.to_csv('sample_dt.csv')
print("=======================================================================================")
print("quantum sample data have been stored into: sample_dt.csv")
print("all_sample_times  :", all_sample_times)
print("all_sample_amplify:", all_sample_amplify)
print("sample rate is    : ", (np.array(all_sample_times) * np.array(all_sample_amplify)).sum() / len(dt))

def output_compare_result(origin: str, quantum: str, p:str|list, p_v: int|list, g: str, m: str, uni_times: int, agg: str, count):
    q_dt = pd.read_csv(quantum)
    real_dt = pd.read_csv(origin)

    if p is str:
        p = [p]
        p_v = [p_v]

    columns_list = q_dt.columns.tolist()
    columns_list[0] = 'Index'
    q_dt.columns = columns_list
    q_dt.loc[:first_sample_times, 'uid'] = q_dt.loc[:first_sample_times]['Index']
    q_dt = q_dt.drop_duplicates(subset='uid')

    for i in range(len(p)):
        if args.op == '=':
            q_dt = q_dt[q_dt[p[i]] == p_v[i]]
        elif args.op == '>=':
            q_dt = q_dt[q_dt[p[i]] >= p_v[i]]
        elif args.op == '<':
            q_dt = q_dt[q_dt[p[i]] < p_v[i]]

    # process null value
    for i in count:
        gi_dt = q_dt[q_dt[g] == i]
        notnull = len(gi_dt[gi_dt[m].notnull()])
        whole = len(gi_dt)
        if whole != 0:
            count[i] = count[i] * notnull / whole

    universe_dt = real_dt.sample(uni_times, replace=False)
    for i in range(len(p)):
        if args.op == '=':
            universe_dt = universe_dt[universe_dt[p[i]] == p_v[i]]
        elif args.op == '>=':
            universe_dt = universe_dt[universe_dt[p[i]] >= p_v[i]]
        elif args.op == '<':
            universe_dt = universe_dt[universe_dt[p[i]] < p_v[i]]
    
    ss_dt = real_dt.groupby(g, group_keys=False).apply(lambda x: x.sample(min(int(uni_times / len(real_dt[g].value_counts())), len(x)), replace=False))
    
    for i in range(len(p)):
        if args.op == '=':
            ss_dt = ss_dt[ss_dt[p[i]] == p_v[i]]
        elif args.op == '>=':
            ss_dt = ss_dt[ss_dt[p[i]] >= p_v[i]]
        elif args.op == '<':
            ss_dt = ss_dt[ss_dt[p[i]] < p_v[i]]
    
    real_calc_dt = real_dt.copy()
    for i in range(len(p)):
        if args.op == '=':
            real_calc_dt = real_calc_dt[real_calc_dt[p[i]] == p_v[i]]
        elif args.op == '>=':
            real_calc_dt = real_calc_dt[real_calc_dt[p[i]] >= p_v[i]]
        elif args.op == '<':
            real_calc_dt = real_calc_dt[real_calc_dt[p[i]] < p_v[i]]

    def histogram(dt: pd.DataFrame, g: str, m: str, agg: str):
        v = dt[[g, m]].groupby(g).agg({m: agg})
        v_aligned = v.reindex(real_dt[[g, m]].groupby(g).agg({m: agg}).index, fill_value=0)
        return v_aligned.to_numpy()[:,0]
    
    real_distribution = histogram(real_calc_dt, g, m, agg)
    q_distribution = histogram(q_dt, g, m, agg)
    universe_distribution = histogram(universe_dt, g, m, agg)
    ss_distribution = histogram(ss_dt, g, m, agg)

    if agg == 'sum':
        q_distribution = histogram(q_dt, g, m, 'mean')
        idx = real_dt[[g, m]].groupby(g).agg({m: agg}).index
        for i in count:
            real_idx = idx.get_loc(i)
            q_distribution[real_idx] *= count[i] * len(real_dt)
    
    if agg == 'count':
        idx = real_dt[[g, m]].groupby(g).agg({m: agg}).index
        for i in count:
            real_idx = idx.get_loc(i)
            q_distribution[real_idx] = count[i] * len(real_dt)
    
    if agg != 'mean':
        universe_distribution = universe_distribution * len(real_dt) / uni_times
        idx = real_dt[[g, m]].groupby(g).agg({m: agg}).index
        for i in count:
            real_idx = idx.get_loc(i)
            ss_distribution[real_idx] = ss_distribution[real_idx] / min(int(uni_times / len(real_dt[g].value_counts())), len(real_dt[real_dt[g] == i])) * len(real_dt[real_dt[g] == i])

    non_zero_mask = real_distribution != 0

    return [
        np.mean(np.abs(real_distribution[non_zero_mask] - q_distribution[non_zero_mask]) / real_distribution[non_zero_mask]),
        np.mean(np.abs(real_distribution[non_zero_mask] - universe_distribution[non_zero_mask]) / real_distribution[non_zero_mask]),
        np.mean(np.abs(real_distribution[non_zero_mask] - ss_distribution[non_zero_mask]) / real_distribution[non_zero_mask]),
        real_distribution[non_zero_mask],
    ]

for i in ['mean', 'count', 'sum']:
    res = output_compare_result(origin=args.dataset,
                quantum='sample_dt.csv',
                p=[P],
                p_v=[P_V],
                g=G,
                m=M,
                uni_times=(np.array(all_sample_times) * np.array(all_sample_amplify)).sum(),
                count=all_group_prob,
                agg=i)

    print("=======================================================================================")
    print("Query: select {}, {}({}) from dataset where {} {} {} group by {}".format(G, i, M, P, args.op, P_V, G))
    print("real answer({}): ".format(i), res[3])
    print("Amplifier          ({}) avg error: ".format(i), res[0])
    print("universe Sampling  ({}) avg error: ".format(i), res[1])
    print("Stratified Sampling({}) avg error: ".format(i), res[2])
    print("=======================================================================================")
