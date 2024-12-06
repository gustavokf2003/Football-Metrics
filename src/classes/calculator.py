import numpy as np

class Calculator:
    """
    A utility class for various calculations related to player positions, velocities, and metrics in a game.
    """

    @staticmethod
    def calculate_average_metrics(metrics: dict) -> dict:
        """
        Calculates the average of metrics for each team.

        Args:
            metrics (dict): Dictionary containing data for each team.

        Returns:
            dict: A dictionary with the average centroid, amplitude, and depth for each team.
        """
        averages = {}

        # Calcula a média de cada métrica para cada equipe
        for team, data in metrics.items():
            centroid_avg = np.mean([centroid[0] / 100 for centroid in data['Centroide']], axis=0)
            amplitude_avg = np.mean(data['Amplitude']) / 100
            depth_avg = np.mean(data['Profundidade']) / 100
            
            averages[team] = {
                'Centroide': centroid_avg,
                'Amplitude': amplitude_avg,
                'Profundidade': depth_avg,
                'Posse': 0
            }

        # Calcula a média de Posse de bola para cada equipe
        averages[1]['Posse'] = metrics[1]['Posse'] / (metrics[1]['Posse'] + metrics[2]['Posse']) * 100
        averages[2]['Posse'] = metrics[2]['Posse'] / (metrics[1]['Posse'] + metrics[2]['Posse']) * 100

        return averages
    
    @staticmethod
    def calculate_average_speeds(players_velocities: dict, team_players: dict) -> dict:
        """
        Calculates the average speed for each player and the average per team.

        Args:
            players_velocities (dict): Dictionary in the format {id: [speeds]}.
            team_players (dict): Dictionary in the format {team: set(ids)}, where `team` is the team number (1 or 2)
                                 and `ids` is a set of player IDs for that team.

        Returns:
            dict: A dictionary with individual and team averages in the format:
                  {
                      "individual_avg": {id: (team, value), ...},
                      "team_avg": {team: value, ...}
                  }
        """
        individual_averages = {}
        team_averages = {1: [], 2: []}  

        # Calcula a média de velocidade para cada jogador
        for player_id, speeds in players_velocities.items():
            individual_avg = sum(speeds) / len(speeds) if speeds else 0
            individual_averages[player_id] = None  

            # Checa a qual equipe o jogador pertence e adiciona a média à lista de médias da equipe
            for team, players in team_players.items():
                if player_id in players:
                    individual_averages[player_id] = (team, individual_avg)
                    team_averages[team].append(individual_avg)
                    break  

        # Calcula a média de velocidade para cada equipe
        team_averages = {team: (sum(speeds) / len(speeds) if speeds else 0)
                         for team, speeds in team_averages.items()}

        return {
            "media_individual": individual_averages,
            "media_equipe": team_averages
        }


    @staticmethod
    def calculate_centroid(positions: list) -> np.ndarray:
        """
        Calculates the centroid of a given set of positions.

        Args:
            positions (list): List of player positions as (x, y) tuples.

        Returns:
            numpy.ndarray: The centroid as an ndarray [[x, y]], or None if no positions are provided.
        """
        if len(positions) > 0:
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            centroid_x = np.mean(x_coords)
            centroid_y = np.mean(y_coords)
            return np.array([[centroid_x, centroid_y]])
        return None

    @staticmethod
    def nearest_player_to_ball(ball_position: np.ndarray, player_positions: np.ndarray, all_detections) -> tuple:
        """
        Finds the nearest player to the ball.

        Args:
            ball_position (numpy.ndarray): Ball position as [[x, y]].
            player_positions (numpy.ndarray): Player positions as [[x1, y1], [x2, y2], ...].
            all_detections: Object containing player IDs and their positions.

        Returns:
            int: ID of the nearest player to the ball.
            float: Distance to the nearest player.
        """
        if len(ball_position) == 0:
            return None, None

        ball = ball_position[0]
        min_distance = float('inf')
        nearest_player_id = None

        for i in range(1, 11):
            position = player_positions[all_detections.tracker_id == i]
            if len(position) == 0:
                continue
            distance = np.sqrt((position[0][0] - ball[0])**2 + (position[0][1] - ball[1])**2)

            if distance < min_distance:
                min_distance = distance
                nearest_player_id = i

        return nearest_player_id, min_distance

    @staticmethod
    def calculate_velocity(current_positions: np.ndarray, previous_positions: np.ndarray, all_detections, previous_all_detections) -> dict:
        """
        Calculates the velocity of each player.

        Args:
            current_positions (numpy.ndarray): Current player positions [[x1, y1], ...].
            previous_positions (numpy.ndarray): Previous player positions [[x1, y1], ...].
            all_detections: Object containing player IDs and their positions.

        Returns:
            dict: A dictionary of velocities for each player.
        """
        velocities = {}
        for i in range(1, 11):  # IDs from 1 to 10
            # Filter current and previous positions for player ID `i`
            current_idx = np.where(all_detections.tracker_id == i)[0]
            previous_idx = np.where(previous_all_detections.tracker_id == i)[0]
            
            if len(current_idx) == 0 or len(previous_idx) == 0:
                continue  # Skip if the player was not detected in the current or previous frame

            # Ensure that the indices do not exceed the bounds
            if current_idx[0] >= len(current_positions) or previous_idx[0] >= len(previous_positions):
                continue

            current = current_positions[current_idx[0]]
            previous = previous_positions[previous_idx[0]]

            # Calculate the distance and velocity
            distance = np.sqrt(
                (current[0] / 100 - previous[0] / 100) ** 2 +
                (current[1] / 100 - previous[1] / 100) ** 2
            )
            velocities[i] = distance * 3.6  # Velocity in km/h

        return velocities

    @staticmethod
    def calculate_amplitude_and_depth(player_positions: np.ndarray) -> np.ndarray:
        """
        Calculates the amplitude and depth based on player positions.

        Args:
            player_positions (numpy.ndarray): Array of player positions as [[x1, y1], [x2, y2], ...].

        Returns:
            numpy.ndarray: Array containing amplitude and depth [[amplitude, depth]].
        """
        if player_positions.size == 0:
            return np.array([[0.0, 0.0]])

        x_min = player_positions[:, 0].min()
        x_max = player_positions[:, 0].max()
        amplitude = x_max - x_min

        y_min = player_positions[:, 1].min()
        y_max = player_positions[:, 1].max()
        depth = y_max - y_min

        return np.array([[amplitude, depth]])
    
    @staticmethod
    def calculate_shannon_entropy_for_players(player_positions_dict, field_size=(3000, 4600), bins=(25, 25)):
        """
        Calculate Shannon entropy for each player based on their positions on a field.

        Args:
            player_positions_dict (dict): Dictionary where the key is the player ID, 
                                        and the value is a list of 2D numpy arrays [[[x1, y1]], [[x2, y2]], ...]
                                        representing the player's positions.
            field_size (tuple): Size of the field (width, height) in meters.
            bins (tuple): Number of bins for the histogram (x_bins, y_bins).

        Returns:
            dict: A dictionary where the key is the player ID, and the value is their Shannon entropy.
        """
        entropy_dict = {}

        for player_id, player_positions in player_positions_dict.items():
            # Filtrar posições não vazias
            valid_positions = [pos for pos in player_positions if pos.size > 0]

            if len(valid_positions) == 0:
                entropy_dict[player_id] = 0  # Caso todas as posições sejam vazias, a entropia é zero
                continue

            # Concatenar todas as posições válidas em um único array
            positions = np.vstack(valid_positions).reshape(-1, 2)

            # Separar coordenadas X e Y
            x_positions = positions[:, 0]
            y_positions = positions[:, 1]

            # Criar histograma 2D das posições
            histogram, _, _ = np.histogram2d(
                x_positions, y_positions,
                bins=bins,
                range=[[0, field_size[0]], [0, field_size[1]]]
            )

            # Achatar o histograma para 1D
            histogram = histogram.flatten()

            # Número total de células
            total_cells = np.sum(histogram)

            if total_cells == 0:
                entropy_dict[player_id] = 0  # Sem dados, entropia é 0
                continue

            # Calcular a função de massa de probabilidade (p_i)
            probabilities = histogram / total_cells

            # Remover probabilidades zero para evitar log(0)
            probabilities = probabilities[probabilities > 0]

            # Calcular a entropia
            entropy = -np.sum(probabilities * np.log2(probabilities))

            # Armazenar a entropia no dicionário
            entropy_dict[player_id] = float(entropy)

        return entropy_dict
