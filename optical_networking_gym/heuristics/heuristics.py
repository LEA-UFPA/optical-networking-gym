import math
import copy
import logging
from typing import Optional
import numpy as np
from optical_networking_gym.topology import Modulation, Path
from optical_networking_gym.utils import rle, link_shannon_entropy_, fragmentation_route_cuts, fragmentation_route_rss
from optical_networking_gym.core.osnr import calculate_osnr
from optical_networking_gym.envs.qrmsa import QRMSAEnv
from gymnasium import Env

from gymnasium import Env
from optical_networking_gym.envs.qrmsa import QRMSAEnv

def get_qrmsa_env(env: Env) -> QRMSAEnv:
    """
    Percorre os wrappers do ambiente até encontrar a instância base de QRMSAEnv.

    Args:
        env (gym.Env): O ambiente potencialmente envolvido em múltiplos wrappers.

    Returns:
        QRMSAEnv: A instância base de QRMSAEnv.

    Raises:
        ValueError: Se QRMSAEnv não for encontrado na cadeia de wrappers.
    """
    while not isinstance(env, QRMSAEnv):
        if hasattr(env, 'env'):
            env = env.env
        else:
            raise ValueError("QRMSAEnv não foi encontrado na cadeia de wrappers do ambiente.")
    return env


def get_action_index(env: QRMSAEnv, path_index: int, modulation_index: int, initial_slot: int) -> int:
    """
    Converte (path_index, modulation_index, initial_slot) em um índice de ação inteiro.
    
    Args:
        env (QRMSAEnv): O ambiente QRMSAEnv.
        path_index (int): Índice da rota.
        modulation_index (int): Índice absoluto da modulação.
        initial_slot (int): Slot inicial para alocação.
    
    Returns:
        int: Índice da ação correspondente.
    """
    # Converter o índice absoluto da modulação para o relativo
    relative_modulation_index = env.max_modulation_idx - modulation_index
    
    return (path_index * env.modulations_to_consider * env.num_spectrum_resources +
            relative_modulation_index * env.num_spectrum_resources +
            initial_slot)

# def decimal_to_array(env: QRMSAEnv, decimal: int, max_values: list[int] = None) -> list[int]:
#     if max_values is None:
#         max_values = [env.k_paths, len(env.modulations), env.num_spectrum_resources]
    
#     array = []
#     for max_val in reversed(max_values):
#         array.insert(0, decimal % max_val)
#         decimal //= max_val
#     print(f"Decimal converted to array {array}")

#     # Mapeia o índice relativo de modulação para o índice absoluto
#     allowed_mods = list(range(env.max_modulation_idx, env.max_modulation_idx - env.modulations_to_consider, -1))
    
#     # O mapeamento de modulação acontece para o segundo índice (array[1])
#     array[1] = allowed_mods[array[1]]
#     print(f"Modulation index mapped to absolute index: {array}")
#     return array



def heuristic_from_mask(env: Env, mask: np.ndarray) -> int:
    print("========== Iniciando heuristic_from_mask ==========")
    total_actions = len(mask)
    print("Total de ações a testar:", total_actions)

    valid_actions = []
    errors = []

    for action_index in range(total_actions):
        # Ação de rejeição deve ser sempre válida na máscara
        if action_index // 31 == 0:
            print("a")

        if action_index == env.action_space.n - 1:
            assert mask[action_index] == 1, "Erro: Ação de rejeição deve ser sempre válida"
            continue

        # Decodifica a ação e imprime os detalhes
        decoded = env.encoded_decimal_to_array(action_index)
        print(action_index,": Decodificação da ação:", decoded)
        path_index, modulation_index, candidate_index = decoded

        # Cria o ambiente de simulação (supondo que get_qrmsa_env retorne uma instância fresca)
        qrmsa_env = get_qrmsa_env(env)
        
        # Seleciona o caminho de acordo com os k-shortest paths
        path = qrmsa_env.k_shortest_paths[
            qrmsa_env.current_service.source,
            qrmsa_env.current_service.destination
        ][path_index]

        # Verifica se a modulação obtida é a esperada
        modulation = qrmsa_env.modulations[modulation_index]
        interval = action_index // 320
        print("*" * 30)
        # Define as modulações esperadas para os dois casos:
        if env.max_modulation_idx > 1:
            # Caso padrão: usamos a modulação de índice max_modulation_idx (a "maior") para intervalos pares
            # e a modulação imediatamente inferior (índice max_modulation_idx - 1) para intervalos ímpares.
            higher_mod = qrmsa_env.modulations[env.max_modulation_idx]
            lower_mod  = qrmsa_env.modulations[env.max_modulation_idx - 1]
        else:
            # Se max_modulation_idx é 0, isto indica que a maior modulação permitida é a primeira da lista (por exemplo, 8QAM).
            # Então, para intervalos pares usamos a próxima modulação (índice 1, ex: 16QAM) e para ímpares a própria.
            higher_mod = qrmsa_env.modulations[1]
            lower_mod  = qrmsa_env.modulations[0]

        # print(f"Expected modulations:\n- Para intervalo par: {higher_mod}\n- Para intervalo ímpar: {lower_mod}")
        # print("*" * 30)

        # Seleciona a modulação esperada com base no valor de 'interval'
        if interval % 2 == 0:
            expected_modulation = higher_mod
        else:
            expected_modulation = lower_mod

        assert modulation == expected_modulation, (
            f"Erro: A modulação {modulation.name} não é a esperada {expected_modulation.name} para o índice {modulation_index}."
        )

        # Determina o número de slots, slots disponíveis e os candidatos válidos
        number_slots = qrmsa_env.get_number_slots(qrmsa_env.current_service, modulation)
        available_slots = qrmsa_env.get_available_slots(path)
        valid_starts = qrmsa_env._get_candidates(available_slots, number_slots, env.num_spectrum_resources)
        
        resource_valid = candidate_index in valid_starts
        if not resource_valid:
            assert mask[action_index] == 0, (
                f"Erro: Ação {action_index} bloqueada por recurso, mas marcada como válida na máscara."
            )
            continue

        # Configura os parâmetros do serviço atual para a simulação
        service = qrmsa_env.current_service
        service.path = path
        service.initial_slot = candidate_index
        service.number_slots = number_slots
        service.center_frequency = (
            qrmsa_env.frequency_start +
            (qrmsa_env.frequency_slot_bandwidth * candidate_index) +
            (qrmsa_env.frequency_slot_bandwidth * (number_slots / 2))
        )
        service.bandwidth = qrmsa_env.frequency_slot_bandwidth * number_slots
        service.launch_power = qrmsa_env.launch_power

        # Calcula o OSNR e valida
        osnr, _, _ = calculate_osnr(qrmsa_env, service)
        # print(f"OSNR calculado para ação {action_index}: {osnr:.2f}")

        threshold = modulation.minimum_osnr + qrmsa_env.margin
        osnr_valid = osnr >= threshold
        if not osnr_valid:
            assert mask[action_index] == 0, (
                f"Erro: Ação {action_index} bloqueada por OSNR, mas marcada como válida na máscara."
            )
            continue

        # Caso a ação seja considerada válida pela simulação, registra a ação
        if mask[action_index] == 1 and not (resource_valid and osnr_valid):
            error_msg = (
                f"Erro: Ação {action_index} marcada como válida na máscara, mas simulação indica ação inválida "
                f"(Resource valid: {resource_valid}, OSNR valid: {osnr_valid}, OSNR: {osnr:.2f})."
            )
            print(error_msg)
            errors.append(error_msg)
        elif mask[action_index] == 0 and (resource_valid and osnr_valid):
            error_msg = (
                f"Erro: Ação {action_index} marcada como inválida na máscara, mas simulação indica ação válida "
                f"(Resource valid: {resource_valid}, OSNR valid: {osnr_valid}, OSNR: {osnr:.2f})."
            )
            print(error_msg)
            errors.append(error_msg)

        valid_actions.append(action_index)
        print(f"Ação {action_index} é considerada válida.")

    print("\n========== Resumo da validação ==========")
    if errors:
        print("Foram encontrados os seguintes erros:")
        for err in errors:
            print(" -", err)
    selected_action = np.random.choice(total_actions)
    return selected_action



def heuristic_load_balancing_first_fit(env: Env) -> tuple[int, bool, bool]:
    """
    Implementa a heurística de balanceamento de carga de forma segura e robusta.

    Esta heurística primeiro ordena as rotas candidatas pela menor carga e depois
    procura a primeira alocação válida, retornando-a imediatamente para evitar
    inconsistências com o ambiente de simulação.
    """
    sim_env = get_qrmsa_env(env)
    current_service = sim_env.current_service
    
    # --- Lógica de Balanceamento de Carga ---
    # 1. Obter todos os caminhos candidatos e os seus índices originais
    candidate_paths_with_indices = list(enumerate(sim_env.k_shortest_paths[current_service.source, current_service.destination]))

    # 2. Calcular a carga para cada caminho
    path_loads = []
    if candidate_paths_with_indices:
        for path_idx, path in candidate_paths_with_indices:
            available_slots = sim_env.get_available_slots(path)
            # Métrica de carga: percentagem de slots ocupados
            load = np.sum(available_slots == 0) / len(available_slots) if len(available_slots) > 0 else 1.0
            path_loads.append((load, path_idx, path))

        # 3. Ordenar os caminhos pela menor carga
        path_loads.sort()
    # --- Fim da Lógica de Balanceamento de Carga ---

    # Itera pela lista de caminhos já ordenada pela menor carga
    for load, path_idx, path in path_loads:
        
        # Itera pelas modulações (da melhor para a pior)
        for mod_idx in range(sim_env.max_modulation_idx, -1, -1):
            modulation = sim_env.modulations[mod_idx]
            required_slots = sim_env.get_number_slots(current_service, modulation)
            
            if required_slots <= 0:
                continue

            available_slots = sim_env.get_available_slots(path)
            valid_starts = sim_env._get_candidates(available_slots, required_slots, sim_env.num_spectrum_resources)
            
            if valid_starts:
                start_slot = valid_starts[0]
                
                # Prepara uma cópia do serviço para a verificação segura de OSNR
                service_copy = copy.deepcopy(current_service)
                service_copy.path = path
                service_copy.initial_slot = start_slot
                service_copy.number_slots = required_slots
                service_copy.current_modulation = modulation
                service_copy.bandwidth = sim_env.frequency_slot_bandwidth * required_slots
                service_copy.launch_power = sim_env.launch_power
                service_copy.center_frequency = (
                    sim_env.frequency_start +
                    (sim_env.frequency_slot_bandwidth * start_slot) +
                    (sim_env.frequency_slot_bandwidth * (required_slots / 2))
                )
                
                osnr, _, _ = calculate_osnr(sim_env, service_copy)
                
                if osnr >= modulation.minimum_osnr + sim_env.margin:
                    # Solução encontrada! Retorna imediatamente.
                    action_index = get_action_index(sim_env, path_idx, mod_idx, start_slot)
                    return action_index, False, False
    
    # Se, após todos os laços, nenhuma solução válida foi encontrada, rejeita o serviço.
    return sim_env.action_space.n - 1, True, False 


def heuristic_highest_snr(env: Env) -> int:
    best_osnr = -np.inf
    best_action = None
    any_blocked_resources = False
    any_blocked_osnr = False

    sim_env = get_qrmsa_env(env)
    source = sim_env.current_service.source
    destination = sim_env.current_service.destination
    k_paths = sim_env.k_shortest_paths[source, destination]

    for path_idx, path in enumerate(k_paths):
        for modulation_idx in range(sim_env.max_modulation_idx, -1, -1):
            modulation = sim_env.modulations[modulation_idx]

            required_slots = sim_env.get_number_slots(sim_env.current_service, modulation)
            if required_slots <= 0:
                continue

            available_slots = sim_env.get_available_slots(path)
            valid_starts = sim_env._get_candidates(available_slots, required_slots, sim_env.num_spectrum_resources)

            if not valid_starts:
                any_blocked_resources = True
                continue

            for candidate_start in valid_starts:
                service = sim_env.current_service
                service.path = path
                service.initial_slot = candidate_start
                service.number_slots = required_slots
                service.current_modulation = modulation
                service.center_frequency = (
                    sim_env.frequency_start +
                    (sim_env.frequency_slot_bandwidth * candidate_start) +
                    (sim_env.frequency_slot_bandwidth * (required_slots / 2))
                )
                service.bandwidth = sim_env.frequency_slot_bandwidth * required_slots
                service.launch_power = sim_env.launch_power

                osnr, _, _ = calculate_osnr(sim_env, service)
                threshold = modulation.minimum_osnr + sim_env.margin

                if osnr >= threshold:
                    if osnr > best_osnr or (osnr == best_osnr and best_action is None):
                        action_index = get_action_index(sim_env, path_idx, modulation_idx, candidate_start)
                        best_osnr = osnr
                        best_action = action_index
                else:
                    any_blocked_osnr = True

    if best_action is not None:
        return best_action, False, False 
    else:
        if any_blocked_osnr:
            any_blocked_resources = False
        return env.action_space.n - 1, any_blocked_resources, any_blocked_osnr

def heuristic_lowest_fragmentation(env: Env) -> tuple[int, bool, bool]:
    sim_env = get_qrmsa_env(env)
    service = sim_env.current_service
    source, destination = service.source, service.destination
    k_paths = sim_env.k_shortest_paths[source, destination]

    best_score = math.inf
    best_action = None
    any_blocked_resources = False
    any_blocked_osnr       = False

    for path_idx, path in enumerate(k_paths):
        raw = sim_env._get_spectrum_slots(path_idx)

        if isinstance(raw, list):
            base_spectra = np.stack(raw, axis=0)
        else:
            arr = np.array(raw)
            if arr.ndim == 1:
                base_spectra = arr[None, :]    # (1, n_slots)
            elif arr.ndim == 2:
                base_spectra = arr
            else:
                raise ValueError(f"Formato inesperado em _get_spectrum_slots: ndim={arr.ndim}")

        for mod_idx in range(sim_env.max_modulation_idx, -1, -1):
            modulation = sim_env.modulations[mod_idx]
            req_slots  = sim_env.get_number_slots(service, modulation)+1
            if req_slots <= 0:
                continue

            available = sim_env.get_available_slots(path)
            candidates = sim_env._get_candidates(
                available, req_slots, sim_env.num_spectrum_resources
            )
            if not candidates:
                any_blocked_resources = True
                continue

            for start in candidates:
                # 3) simula a alocação: clone e pinta o bloco como 1=ocupado
                tmp = base_spectra.copy()
                tmp[:, start:start+req_slots] = 1

                # 4) calcula métricas de fragmentação
                # 4a) entropia média
                entropies = [ link_shannon_entropy_(row.tolist()) for row in tmp ]
                se   = sum(entropies) / len(entropies) if entropies else 0.0
                # 4b) cuts
                cuts = fragmentation_route_cuts([row.tolist() for row in tmp])
                # 4c) rss
                rss  = fragmentation_route_rss([row.tolist() for row in tmp])

                score = 0.33 * se + 0.33 * cuts + 0.34 * rss

                # 5) verifica OSNR
                service.path             = path
                service.initial_slot     = start
                service.number_slots     = req_slots
                service.current_modulation = modulation
                service.center_frequency = (
                    sim_env.frequency_start
                    + sim_env.frequency_slot_bandwidth * start
                    + sim_env.frequency_slot_bandwidth * (req_slots / 2)
                )
                service.bandwidth    = sim_env.frequency_slot_bandwidth * req_slots
                service.launch_power = sim_env.launch_power

                osnr, _, _ = calculate_osnr(sim_env, service)
                if osnr < modulation.minimum_osnr + sim_env.margin:
                    any_blocked_osnr = True
                    continue

                # 6) escolhe o candidato de menor fragmentação
                if score < best_score:
                    best_score  = score
                    best_action = get_action_index(sim_env, path_idx, mod_idx, start)

    # 7) retorna conforme convenção
    if best_action is not None:
        return best_action, False, False
    else:
        if any_blocked_osnr:
            any_blocked_resources = False
        return env.action_space.n - 1, any_blocked_resources, any_blocked_osnr




def shortest_available_path_first_fit_best_modulation(
    mask: np.ndarray,
) -> Optional[int]:
    return int(np.where(mask == 1)[0][0])

def rnd(
    mask: np.ndarray,
) -> Optional[int]:
    valid_actions = np.where(mask == 1)[0]
    return int(np.random.choice(valid_actions))


def shortest_available_path_lowest_spectrum_best_modulation(env: Env) -> tuple[int, bool, bool]:
    """
    Seleciona a rota mais curta disponível com o menor índice de espectro e a melhor modulação.
    """
    sim_env = get_qrmsa_env(env)
    blocked_due_to_resources, blocked_due_to_osnr = False, False

    # Itera pelos caminhos (k-shortest paths)
    for path_idx, path in enumerate(sim_env.k_shortest_paths[
        sim_env.current_service.source,
        sim_env.current_service.destination,
    ]):
        # Itera pelas modulações, da melhor para a pior
        for modulation_idx in range(sim_env.max_modulation_idx, -1, -1):
            modulation = sim_env.modulations[modulation_idx]

            # 1. Pega o número de slots necessários (SEM adicionar guard bands manualmente)
            required_slots = sim_env.get_number_slots(sim_env.current_service, modulation)
            if required_slots <= 0:
                continue

            # 2. Usa o método do ambiente para obter os slots candidatos
            available_slots = sim_env.get_available_slots(path)
            valid_starts = sim_env._get_candidates(available_slots, required_slots, sim_env.num_spectrum_resources)

            # Se não houver slots válidos, continue para a próxima modulação/caminho
            if not valid_starts:
                blocked_due_to_resources = True
                continue
            
            # 3. Pega o primeiro candidato (que corresponde ao menor espectro)
            candidate_start = valid_starts[0]

            # 4. Configura o serviço e verifica o OSNR
            service = sim_env.current_service
            service.path = path
            service.initial_slot = candidate_start
            service.number_slots = required_slots
            # ... (configure os outros parâmetros do serviço como antes) ...
            service.center_frequency = (
                        sim_env.frequency_start +
                        (sim_env.frequency_slot_bandwidth * candidate_start) +
                        (sim_env.frequency_slot_bandwidth * (required_slots / 2)))
            service.bandwidth = sim_env.frequency_slot_bandwidth * required_slots
            service.launch_power = sim_env.launch_power

            osnr, _, _ = calculate_osnr(sim_env, service)
            threshold = modulation.minimum_osnr + sim_env.margin

            # 5. Se o OSNR for válido, retorne a ação
            if osnr >= threshold:
                action_index = get_action_index(sim_env, path_idx, modulation_idx, candidate_start)
                return action_index, False, False
            else:
                blocked_due_to_osnr = True

    # 6. Se nenhum candidato válido for encontrado, retorne a ação de rejeição
    if blocked_due_to_osnr:
        blocked_due_to_resources = False
    return env.action_space.n - 1, blocked_due_to_resources, blocked_due_to_osnr
def best_modulation_load_balancing(
    env: Env,
) -> Optional[int]:
    """
    Balanceia a carga selecionando a melhor modulação e minimizando a carga na rota.
    
    Args:
        env (gym.Env): O ambiente potencialmente envolvido em wrappers.
    
    Returns:
        Optional[int]: Índice da ação correspondente, ou a ação de rejeição se permitido, ou None.
    """
    qrmsa_env: QRMSAEnv = get_qrmsa_env(env)  # Descompactar o ambiente
    solution = None
    lowest_load = float('inf')

    for idm, modulation in zip(
        range(len(qrmsa_env.modulations) - 1, -1, -1),
        reversed(qrmsa_env.modulations)
    ):
        for idp, path in enumerate(qrmsa_env.k_shortest_paths[
            qrmsa_env.current_service.source,
            qrmsa_env.current_service.destination,
        ]):
            available_slots = qrmsa_env.get_available_slots(path)
            number_slots = qrmsa_env.get_number_slots(qrmsa_env.current_service, modulation)  # +2 para guard band

            initial_indices, values, lengths = rle(available_slots)
            sufficient_indices = np.where(lengths >= number_slots+1)
            available_indices = np.where(values == 1)
            final_indices = np.intersect1d(available_indices, sufficient_indices)

            if final_indices.size > 0:
                qrmsa_env.current_service.blocked_due_to_resources = False
                initial_slot = initial_indices[final_indices][0]

                # Atualizar parâmetros do serviço
                qrmsa_env.current_service.path = path
                qrmsa_env.current_service.initial_slot = initial_slot
                qrmsa_env.current_service.number_slots = number_slots
                qrmsa_env.current_service.center_frequency = qrmsa_env.frequency_start + \
                    (qrmsa_env.frequency_slot_bandwidth * initial_slot) + \
                    (qrmsa_env.frequency_slot_bandwidth * (number_slots / 2))
                qrmsa_env.current_service.bandwidth = qrmsa_env.frequency_slot_bandwidth * number_slots
                qrmsa_env.current_service.launch_power = qrmsa_env.launch_power

                # Calcular OSNR
                osnr, _, _ = calculate_osnr(qrmsa_env, qrmsa_env.current_service)
                if osnr >= modulation.minimum_osnr + qrmsa_env.margin:
                    # Converter para índice de ação
                    action = get_action_index(qrmsa_env, idp, idm, initial_slot)
                    return action, False, False  # ou uma ação padrão específica

    # Se nenhuma ação válida encontrada, retornar a ação de rejeição se permitido
    return qrmsa_env.reject_action, False, False

def load_balancing_best_modulation(env: Env) -> tuple[int, bool, bool]:
    """
    Balanceia a carga selecionando a rota com menor uso e, nela, a melhor modulação possível.
    A seleção do slot segue a política 'first-fit'.
    """
    sim_env = get_qrmsa_env(env)
    solution = None
    lowest_load = float('inf')
    
    # Flags para rastrear o motivo do bloqueio, caso ocorra
    any_blocked_resources = False
    any_blocked_osnr = False

    # Itera pelos caminhos (k-shortest paths)
    for path_idx, path in enumerate(sim_env.k_shortest_paths[
        sim_env.current_service.source,
        sim_env.current_service.destination,
    ]):
        available_slots_on_path = sim_env.get_available_slots(path)
        
        # Calcula a carga atual da rota
        # Nota: A métrica de carga pode ser ajustada conforme a necessidade.
        # Usando a soma de slots livres como uma aproximação inversa da carga.
        current_load = np.sum(available_slots_on_path == 0) / len(path.links)

        # Se a carga atual não for melhor que a já encontrada, pula para o próximo caminho
        if current_load >= lowest_load:
            continue

        # Itera pelas modulações, da melhor para a pior
        for modulation_idx in range(sim_env.max_modulation_idx, -1, -1):
            modulation = sim_env.modulations[modulation_idx]
            
            # 1. Pega o número de slots (sem adicionar guard bands manualmente)
            required_slots = sim_env.get_number_slots(sim_env.current_service, modulation)
            if required_slots <= 0:
                continue

            # 2. Usa o método do ambiente para obter os slots candidatos
            valid_starts = sim_env._get_candidates(available_slots_on_path, required_slots, sim_env.num_spectrum_resources)

            if not valid_starts:
                any_blocked_resources = True
                continue  # Tenta a próxima modulação (de qualidade inferior)

            # 3. Pega o primeiro candidato (first-fit)
            candidate_start = valid_starts[0]

            # 4. Configura o serviço e verifica o OSNR
            service = sim_env.current_service
            service.path = path
            service.initial_slot = candidate_start
            service.number_slots = required_slots
            service.current_modulation = modulation
            service.center_frequency = (
                        sim_env.frequency_start +
                        (sim_env.frequency_slot_bandwidth * candidate_start) +
                        (sim_env.frequency_slot_bandwidth * (required_slots / 2)))
            service.bandwidth = sim_env.frequency_slot_bandwidth * required_slots
            service.launch_power = sim_env.launch_power

            osnr, _, _ = calculate_osnr(sim_env, service)
            threshold = modulation.minimum_osnr + sim_env.margin

            # 5. Se o OSNR for válido, encontramos a melhor opção para ESTA rota
            if osnr >= threshold:
                lowest_load = current_load
                solution = get_action_index(sim_env, path_idx, modulation_idx, candidate_start)
                # Encontrou a melhor modulação para esta rota, pode ir para a próxima rota
                break 
            else:
                any_blocked_osnr = True

    # 6. Retorna a melhor solução encontrada ou a ação de rejeição
    if solution is not None:
        return solution, False, False
    else:
        # Define a causa principal do bloqueio para fins de log
        if any_blocked_osnr:
            any_blocked_resources = False
        return env.action_space.n - 1, any_blocked_resources, any_blocked_osnr

def _calculate_allocation_possibilities(available_slots: np.ndarray, required_slots: int) -> int:
    """
    Calcula o número total de maneiras que uma requisição pode ser alocada em um espectro.
    """
    if required_slots <= 0:
        return 0
    
    initial_indices, values, lengths = rle(available_slots)
    
    total_possibilities = 0
    for i in range(len(values)):
        if values[i] == 1 and lengths[i] >= required_slots:
            gap_size = lengths[i]
            possibilities_in_gap = gap_size - required_slots + 1
            total_possibilities += possibilities_in_gap
            
    return total_possibilities

def heuristic_mscl(env: Env) -> tuple[int, bool, bool]:
    """
    Heurística MSCL - Versão Otimizada.
    A identificação de rotas interferentes foi movida para fora dos laços internos
    para reduzir a complexidade computacional.
    """
    logging.info("Iniciando cálculo com a heurística MSCL para um novo serviço...")

    sim_env = get_qrmsa_env(env)
    current_service = sim_env.current_service
    source, destination = current_service.source, current_service.destination
    
    best_loss = float('inf')
    best_action = None
    any_blocked_resources = False
    any_blocked_osnr = False

    candidate_paths = sim_env.k_shortest_paths[source, destination]

    # --- OTIMIZAÇÃO: Pré-cálculo das rotas interferentes ---
    all_network_paths = []
    for node_pair in sim_env.k_shortest_paths:
        for p in sim_env.k_shortest_paths[node_pair]:
            all_network_paths.append(p)

    interfering_paths_map = {}
    for i, path in enumerate(candidate_paths):
        interfering_paths_map[i] = []
        path_links = set(path.links)
        for p in all_network_paths:
            if path_links.intersection(p.links):
                interfering_paths_map[i].append(p)
    # --- FIM DA OTIMIZAÇÃO ---

    for path_idx, path in enumerate(candidate_paths):
        # Pega o conjunto pré-calculado de rotas interferentes para esta rota
        interfering_paths = interfering_paths_map[path_idx]

        for mod_idx in range(sim_env.max_modulation_idx, -1, -1):
            modulation = sim_env.modulations[mod_idx]
            required_slots = sim_env.get_number_slots(current_service, modulation)
            
            if required_slots <= 0:
                continue

            available_slots_on_path = sim_env.get_available_slots(path)
            candidate_starts = sim_env._get_candidates(
                available_slots_on_path, required_slots, sim_env.num_spectrum_resources
            )

            if not candidate_starts:
                any_blocked_resources = True
                continue
            
            for start_slot in candidate_starts:
                service_copy = copy.deepcopy(current_service)
                service_copy.path = path
                service_copy.initial_slot = start_slot
                service_copy.number_slots = required_slots
                service_copy.current_modulation = modulation
                service_copy.bandwidth = sim_env.frequency_slot_bandwidth * required_slots
                service_copy.launch_power = sim_env.launch_power
                service_copy.center_frequency = (
                    sim_env.frequency_start +
                    (sim_env.frequency_slot_bandwidth * start_slot) +
                    (sim_env.frequency_slot_bandwidth * (required_slots / 2))
                )
                
                osnr, _, _ = calculate_osnr(sim_env, service_copy)
                
                if osnr < modulation.minimum_osnr + sim_env.margin:
                    any_blocked_osnr = True
                    continue

                total_capacity_loss = 0
                blockage_mask = np.ones(sim_env.num_spectrum_resources, dtype=int)
                blockage_mask[start_slot : start_slot + required_slots] = 0

                possible_bit_rates = sim_env.bit_rates
                for br in possible_bit_rates:
                    hypothetical_service = copy.deepcopy(current_service)
                    hypothetical_service.bit_rate = br
                    n_slots_for_br = sim_env.get_number_slots(hypothetical_service, modulation)

                    for interfering_path in interfering_paths:
                        av_slots_before = sim_env.get_available_slots(interfering_path)
                        s_before = _calculate_allocation_possibilities(av_slots_before, n_slots_for_br)
                        
                        av_slots_after = np.logical_and(av_slots_before, blockage_mask)
                        s_after = _calculate_allocation_possibilities(av_slots_after, n_slots_for_br)
                        
                        total_capacity_loss += (s_before - s_after)

                if total_capacity_loss < best_loss:
                    best_loss = total_capacity_loss
                    best_action = get_action_index(sim_env, path_idx, mod_idx, start_slot)
                    
    if best_action is not None:
        return best_action, False, False
    else:
        if any_blocked_osnr and not any_blocked_resources:
             any_blocked_resources = False
        return sim_env.action_space.n - 1, any_blocked_resources, any_blocked_osnr
    
def _get_largest_contiguous_block(available_slots: np.ndarray) -> int:
    """Função auxiliar para encontrar o maior bloco contíguo de slots livres."""
    if not np.any(available_slots):
        return 0
    
    initial_indices, values, lengths = rle(available_slots)
    
    max_len = 0
    for i in range(len(values)):
        if values[i] == 1 and lengths[i] > max_len:
            max_len = lengths[i]
            
    return max_len

def heuristic_mscl_simplified(env: Env) -> tuple[int, bool, bool]:
    """
    Versão simplificada e otimizada do MSCL para testes rápidos.
    
    Esta versão foi adaptada para ser mais robusta, usando uma política First-Fit
    para a seleção de slots, a fim de evitar inconsistências com o ambiente.
    """
    logging.info("Iniciando cálculo com a heurística MSCL SIMPLIFICADA (Robusta)...")
    
    sim_env = get_qrmsa_env(env)
    current_service = sim_env.current_service
    
    best_fragmentation_score = -1
    best_action = None
    any_blocked_resources = False
    any_blocked_osnr = False

    candidate_paths = sim_env.k_shortest_paths[current_service.source, current_service.destination]

    for path_idx, path in enumerate(candidate_paths):
        for mod_idx in range(sim_env.max_modulation_idx, -1, -1):
            modulation = sim_env.modulations[mod_idx]
            required_slots = sim_env.get_number_slots(current_service, modulation)
            
            if required_slots <= 0:
                continue

            available_slots_on_path = sim_env.get_available_slots(path)
            candidate_starts = sim_env._get_candidates(
                available_slots_on_path, required_slots, sim_env.num_spectrum_resources
            )

            # ALTERAÇÃO: Em vez de iterar por todos os candidatos, usamos apenas o primeiro (First-Fit)
            if not candidate_starts:
                any_blocked_resources = True
                continue
            
            start_slot = candidate_starts[0] # Pega apenas o primeiro candidato
            
            # A verificação de OSNR continua a ser importante
            service_copy = copy.deepcopy(current_service)
            service_copy.path = path
            service_copy.initial_slot = start_slot
            service_copy.number_slots = required_slots
            service_copy.current_modulation = modulation
            service_copy.bandwidth = sim_env.frequency_slot_bandwidth * required_slots
            service_copy.launch_power = sim_env.launch_power
            service_copy.center_frequency = (
                sim_env.frequency_start +
                (sim_env.frequency_slot_bandwidth * start_slot) +
                (sim_env.frequency_slot_bandwidth * (required_slots / 2))
            )
            
            osnr, _, _ = calculate_osnr(sim_env, service_copy)
            
            if osnr >= modulation.minimum_osnr + sim_env.margin:
                # Se a combinação for válida, calcula a sua pontuação de fragmentação
                temp_available_slots = available_slots_on_path.copy()
                temp_available_slots[start_slot : start_slot + required_slots] = 0
                current_score = _get_largest_contiguous_block(temp_available_slots)

                # Compara com a melhor pontuação encontrada até agora
                if current_score > best_fragmentation_score:
                    best_fragmentation_score = current_score
                    best_action = get_action_index(sim_env, path_idx, mod_idx, start_slot)
            else:
                any_blocked_osnr = True
                
    # Retorna a melhor ação encontrada após verificar todas as rotas e modulações
    if best_action is not None:
        logging.info(f"MSCL Simplificado escolheu ação {best_action} com pontuação de fragmentação {best_fragmentation_score}.")
        return best_action, False, False
    else:
        if any_blocked_osnr and not any_blocked_resources:
             any_blocked_resources = False
        return sim_env.action_space.n - 1, any_blocked_resources, any_blocked_osnr
def heuristic_mscl_sequential_simplified(env: Env) -> tuple[int, bool, bool]:
    """
    Implementa a heurística MSCL Sequencial de forma otimizada para testes.

    A heurística itera pelas rotas em sequência. Para a primeira rota que tiver
    recursos, ela encontra a melhor alocação (baseada em fragmentação mínima)
    APENAS NESSA ROTA e retorna a ação imediatamente.
    """
    logging.info("Iniciando cálculo com a heurística MSCL SEQUENCIAL Simplificada...")
    
    sim_env = get_qrmsa_env(env)
    current_service = sim_env.current_service
    
    any_blocked_resources = False
    any_blocked_osnr = False

    # Itera pelas rotas candidatas em sequência
    for path_idx, path in enumerate(sim_env.k_shortest_paths[current_service.source, current_service.destination]):
        
        best_score_for_this_path = -1
        best_action_for_this_path = None
        
        # Para a rota atual, encontra a melhor combinação de modulação e slot
        for mod_idx in range(sim_env.max_modulation_idx, -1, -1):
            modulation = sim_env.modulations[mod_idx]
            required_slots = sim_env.get_number_slots(current_service, modulation)
            
            if required_slots <= 0:
                continue

            available_slots = sim_env.get_available_slots(path)
            candidate_starts = sim_env._get_candidates(available_slots, required_slots, sim_env.num_spectrum_resources)
            
            if not candidate_starts:
                any_blocked_resources = True
                continue

            # Para manter a robustez, usamos o primeiro slot candidato (First-Fit)
            start_slot = candidate_starts[0]
            
            # Verifica o OSNR
            service_copy = copy.deepcopy(current_service)
            # ... (preenche todos os atributos do service_copy como antes)
            service_copy.path = path
            service_copy.initial_slot = start_slot
            service_copy.number_slots = required_slots
            service_copy.current_modulation = modulation
            service_copy.bandwidth = sim_env.frequency_slot_bandwidth * required_slots
            service_copy.launch_power = sim_env.launch_power
            service_copy.center_frequency = (
                sim_env.frequency_start +
                (sim_env.frequency_slot_bandwidth * start_slot) +
                (sim_env.frequency_slot_bandwidth * (required_slots / 2))
            )
            
            osnr, _, _ = calculate_osnr(sim_env, service_copy)
            
            if osnr >= modulation.minimum_osnr + sim_env.margin:
                # Calcula a pontuação de fragmentação para esta opção válida
                temp_slots = available_slots.copy()
                temp_slots[start_slot : start_slot + required_slots] = 0
                current_score = _get_largest_contiguous_block(temp_slots)
                
                # Se esta for a melhor opção encontrada ATÉ AGORA PARA ESTA ROTA
                if current_score > best_score_for_this_path:
                    best_score_for_this_path = current_score
                    best_action_for_this_path = get_action_index(sim_env, path_idx, mod_idx, start_slot)
            else:
                any_blocked_osnr = True

        # FIM DO LAÇO DE MODULAÇÕES
        # Se uma ação válida foi encontrada para esta rota, retorna-a imediatamente.
        if best_action_for_this_path is not None:
            logging.info(f"MSCL Sequencial encontrou a melhor ação {best_action_for_this_path} na rota {path_idx} e está a retornar.")
            return best_action_for_this_path, False, False

    # Se o laço terminar sem nenhuma solução encontrada em nenhuma rota
    logging.warning("MSCL Sequencial não encontrou nenhuma alocação válida em nenhuma rota.")
    if any_blocked_osnr and not any_blocked_resources:
        any_blocked_resources = False
    return sim_env.action_space.n - 1, any_blocked_resources, any_blocked_osnr

def heuristic_shortest_available_path_first_fit_best_modulation(env: Env) -> int:
    blocked_due_to_resources, blocked_due_to_osnr = False, False
    sim_env = get_qrmsa_env(env)
    source = sim_env.current_service.source
    destination = sim_env.current_service.destination
    k_paths = sim_env.k_shortest_paths[source, destination]
    for path_idx, path in enumerate(k_paths):
        for modulation_idx in range(sim_env.max_modulation_idx, -1, -1):
            modulation = sim_env.modulations[modulation_idx]

            required_slots = sim_env.get_number_slots(sim_env.current_service, modulation)
            if required_slots <= 0:
                continue 
            available_slots = sim_env.get_available_slots(path)
            valid_starts = sim_env._get_candidates(available_slots, required_slots, sim_env.num_spectrum_resources)
            
            if not valid_starts:
                blocked_due_to_resources = True
                continue 
            candidate_start = valid_starts[0]
            service = sim_env.current_service
            service.path = path
            service.initial_slot = candidate_start
            service.number_slots = required_slots
            service.current_modulation = modulation
            service.center_frequency = (
                        sim_env.frequency_start +
                        (sim_env.frequency_slot_bandwidth * candidate_start) +
                        (sim_env.frequency_slot_bandwidth * (required_slots / 2)))
            
            service.bandwidth = sim_env.frequency_slot_bandwidth * required_slots
            service.launch_power = sim_env.launch_power
            
            osnr, _, _ = calculate_osnr(sim_env, service)
            threshold = modulation.minimum_osnr + sim_env.margin
            if osnr >= threshold:
                action_index = get_action_index(sim_env, path_idx, modulation_idx, candidate_start)
                return action_index, False, False 
            else:
                blocked_due_to_osnr = True
                if blocked_due_to_resources:
                    blocked_due_to_resources = False
    
    return env.action_space.n - 1, blocked_due_to_resources, blocked_due_to_osnr 
#def heuristic_safe_first_fit(env: Env) -> tuple[int, bool, bool]:
    """
    Esta é a implementação de um molde seguro para heurísticas neste ambiente.

    A sua estabilidade vem da lógica de "retorno imediato": ela encontra a primeira
    combinação válida de rota, modulação e espectro e retorna a ação sem
    explorar outras opções, evitando erros de inconsistência do simulador.
    """
    sim_env = get_qrmsa_env(env)
    current_service = sim_env.current_service
    
    # Itera pelas rotas na ordem padrão (mais curta primeiro)
    for path_idx, path in enumerate(sim_env.k_shortest_paths[current_service.source, current_service.destination]):
        
        # Itera pelas modulações (da melhor para a pior)
        for mod_idx in range(sim_env.max_modulation_idx, -1, -1):
            modulation = sim_env.modulations[mod_idx]
            required_slots = sim_env.get_number_slots(current_service, modulation)
            
            if required_slots <= 0:
                continue

            available_slots = sim_env.get_available_slots(path)
            valid_starts = sim_env._get_candidates(available_slots, required_slots, sim_env.num_spectrum_resources)
            
            if valid_starts:
                start_slot = valid_starts[0]
                
                # Prepara uma cópia segura do serviço para a verificação de OSNR
                service_copy = copy.deepcopy(current_service)
                service_copy.path = path
                service_copy.initial_slot = start_slot
                service_copy.number_slots = required_slots
                service_copy.current_modulation = modulation
                service_copy.bandwidth = sim_env.frequency_slot_bandwidth * required_slots
                service_copy.launch_power = sim_env.launch_power
                service_copy.center_frequency = (
                    sim_env.frequency_start +
                    (sim_env.frequency_slot_bandwidth * start_slot) +
                    (sim_env.frequency_slot_bandwidth * (required_slots / 2))
                )
                
                osnr, _, _ = calculate_osnr(sim_env, service_copy)
                
                if osnr >= modulation.minimum_osnr + sim_env.margin:
                    # Solução encontrada! Retorna imediatamente.
                    action_index = get_action_index(sim_env, path_idx, mod_idx, start_slot)
                    return action_index, False, False
    
    # Se, após todos os laços, nenhuma solução válida foi encontrada, rejeita o serviço.
    return sim_env.action_space.n - 1, True, False

def heuristic_psr(env: Env, variant: str = 'O', coef_dist: float = 1.0, coef_slots: float = 1.0) -> tuple[int, bool, bool]:
    """
    Heurística PSR (Power Series Routing) adaptada para o simulador e para operar em ambiente multibanda.
    variantes: 'C' para PSR-C (clássico, soma), 'O' para PSR-O (otimizado, produto).
    coef_dist, coef_slots: coeficientes para os custos.
    """
    sim_env = get_qrmsa_env(env)
    current_service = sim_env.current_service
    source, destination = current_service.source, current_service.destination
    k_paths = sim_env.k_shortest_paths[source, destination]
    
    best_action = sim_env.action_space.n - 1  # Ação padrão de rejeição
    best_cost = float('inf')
    best_path_idx = None
    best_modulation_idx = None
    best_candidate_start = None
    
    for path_idx, path in enumerate(k_paths):
        path_cost = 0.0 if variant == 'C' else 1.0
        D = max(link.length for link in path.links) if path.links else 1.0  # D é o maior link do caminho
        
        for link in path.links:
            x1 = link.length / D if D > 0 else 1.0
            
            # Obtém o número de slots para a banda atual
            available_slots = sim_env.get_available_slots(path)
            min_mod = sim_env.modulations[-1]
            n_slots = sim_env.get_number_slots(current_service, min_mod)
            
            encaixes = 0
            if n_slots > 0:
                ones = 0
                for s in available_slots:
                    if s == 1:
                        ones += 1
                        if ones >= n_slots:
                            encaixes += 1
                    else:
                        ones = 0
            x2 = 1.0 / (encaixes + 1)
            
            if variant == 'C':
                path_cost += coef_dist * x1 + coef_slots * x2
            else:
                path_cost *= (coef_dist * x1) * (coef_slots * x2)
        
        # Após calcular o custo do caminho, tenta encontrar uma alocação válida
        if path_cost < best_cost:
            for modulation_idx in range(sim_env.max_modulation_idx, -1, -1):
                modulation = sim_env.modulations[modulation_idx]
                required_slots = sim_env.get_number_slots(current_service, modulation)
                
                if required_slots <= 0:
                    continue
                
                available_slots = sim_env.get_available_slots(path)
                valid_starts = sim_env._get_candidates(available_slots, required_slots, sim_env.num_spectrum_resources)
                
                if not valid_starts:
                    continue
                
                for candidate_start in valid_starts:
                    # NOVO: só retorna se o slot está realmente livre
                    if not sim_env.is_path_free(path, candidate_start, required_slots):
                        continue
                    
                    # Prepara uma cópia do serviço para a verificação segura de OSNR
                    service_copy = copy.deepcopy(current_service)
                    service_copy.path = path
                    service_copy.initial_slot = candidate_start
                    service_copy.number_slots = required_slots
                    service_copy.current_modulation = modulation
                    service_copy.bandwidth = sim_env.frequency_slot_bandwidth * required_slots
                    service_copy.launch_power = sim_env.launch_power
                    service_copy.center_frequency = (
                        sim_env.frequency_start +
                        (sim_env.frequency_slot_bandwidth * candidate_start) +
                        (sim_env.frequency_slot_bandwidth * (required_slots / 2))
                    )
                    
                    osnr, _, _ = calculate_osnr(sim_env, service_copy)
                    threshold = modulation.minimum_osnr + sim_env.margin
                    
                    if osnr >= threshold:
                        # Encontrou uma solução válida, atualiza os melhores valores
                        best_cost = path_cost
                        best_path_idx = path_idx
                        best_modulation_idx = modulation_idx
                        best_candidate_start = candidate_start
                        break  # Não precisa testar outros slots para esta modulação
                if best_path_idx is not None:
                    break  # Não precisa testar outras modulações para este caminho
        if best_path_idx is not None:
            break  # Não precisa testar outros caminhos
    
    # Se encontrou uma solução válida, retorna a ação correspondente
    if best_path_idx is not None:
        action_index = get_action_index(sim_env, best_path_idx, best_modulation_idx, best_candidate_start)
        return action_index, False, False
    else:
        return sim_env.action_space.n - 1, True, False
    
def heuristic_exact_fit(env: Env) -> tuple[int, bool, bool]:
    """
    Exact-Fit (EF) heuristic implementation.
    """
    sim_env = get_qrmsa_env(env)
    service = sim_env.current_service
    source, destination = service.source, service.destination
    k_paths = sim_env.k_shortest_paths[source, destination]
    
    any_blocked_resources = False
    any_blocked_osnr = False
    
    for path_idx, path in enumerate(k_paths):
        for modulation_idx in range(sim_env.max_modulation_idx, -1, -1):
            modulation = sim_env.modulations[modulation_idx]
            required_slots = sim_env.get_number_slots(service, modulation)
            
            # Se não há slots suficientes para a modulação atual, pula para próxima
            if required_slots <= 0:
                continue
                
            # Obtém os slots disponíveis no caminho atual
            available_slots = sim_env.get_available_slots(path)
            initial_indices, values, lengths = rle(available_slots)
            
            # Encontra blocos disponíveis (onde values == 1)
            # Retorna índices onde há slots contíguos disponíveis
            available_indices = np.where(values == 1)[0]
            
            # Se não há blocos disponíveis, marca como bloqueado por recursos e continua
            if len(available_indices) == 0:
                any_blocked_resources = True
                continue
            
            # Primeiro, procura por correspondências exatas
            # Verifica se há blocos com exatamente o número de slots necessários
            exact_match_indices = []
            for idx in available_indices:
                if lengths[idx] == required_slots:
                    exact_match_indices.append(idx)
            
            # Se encontrou correspondências exatas, usa a primeira
            # Isso minimiza a fragmentação do espectro
            if exact_match_indices:
                chosen_idx = exact_match_indices[0]
                initial_slot = initial_indices[chosen_idx]
            else:
                # Se não há correspondência exata, aplica heurística alternativa
                # Implementação da heurística Best-Fit: encontra o menor bloco
                # que ainda seja suficiente para acomodar a requisição
                # Isso minimiza o desperdício e reduz fragmentação
                best_fit_idx = None
                best_fit_size = float('inf')
                
                for idx in available_indices:
                    if lengths[idx] >= required_slots and lengths[idx] < best_fit_size:
                        best_fit_size = lengths[idx]
                        best_fit_idx = idx
                
                # Alternativa: Worst-Fit (código original comentado)
                # worst_fit_idx = None
                # worst_fit_size = 0
                # for idx in available_indices:
                #     if lengths[idx] >= required_slots and lengths[idx] > worst_fit_size:
                #         worst_fit_size = lengths[idx]
                #         worst_fit_idx = idx
                
                # Se não encontrou bloco grande o suficiente, marca como bloqueado
                if best_fit_idx is None:
                    any_blocked_resources = True
                    continue
                    
                initial_slot = initial_indices[best_fit_idx]
            
            # Atualiza os parâmetros do serviço com a alocação escolhida
            service.path = path
            service.initial_slot = initial_slot
            service.number_slots = required_slots
            service.current_modulation = modulation
            service.center_frequency = (
                sim_env.frequency_start +
                (sim_env.frequency_slot_bandwidth * initial_slot) +
                (sim_env.frequency_slot_bandwidth * (required_slots / 2))
            )
            service.bandwidth = sim_env.frequency_slot_bandwidth * required_slots
            service.launch_power = sim_env.launch_power
            
            # Calcula OSNR e verifica se atende o limite mínimo
            # OSNR deve ser maior que o mínimo da modulação mais a margem de segurança
            osnr, _, _ = calculate_osnr(sim_env, service)
            threshold = modulation.minimum_osnr + sim_env.margin
            
            if osnr >= threshold:
                # Alocação válida encontrada - retorna o índice da ação correspondente
                action_index = get_action_index(sim_env, path_idx, modulation_idx, initial_slot)
                return action_index, False, False
            else:
                # Marca como bloqueado por OSNR insuficiente
                any_blocked_osnr = True
    
    # Nenhuma alocação válida encontrada
    # Se houve bloqueio por OSNR, considera apenas este bloqueio
    if any_blocked_osnr:
        any_blocked_resources = False
    
    # Retorna ação de rejeição e flags de bloqueio
    return env.action_space.n - 1, any_blocked_resources, any_blocked_osnr