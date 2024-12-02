import numpy as np

class Calculator:
    """
    A utility class for various calculations related to player positions, velocities, and metrics in a game.
    """

    @staticmethod
    def calculate_average_metrics(metrics):
        """
        Calculates the average of metrics for each team.

        Args:
            metrics (dict): Dictionary containing data for each team.

        Returns:
            dict: A dictionary with the average centroid, amplitude, and depth for each team.
        """
        averages = {}
        for team, data in metrics.items():
            centroid_avg = np.mean([centroid[0] for centroid in data['centroid']], axis=0)
            amplitude_avg = np.mean(data['amplitude'])
            depth_avg = np.mean(data['depth'])
            
            averages[team] = {
                'centroid': centroid_avg,
                'amplitude': amplitude_avg,
                'depth': depth_avg
            }
        return averages

    @staticmethod
    def calculate_centroid(positions):
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
    def nearest_player_to_ball(ball_position, player_positions, all_detections):
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
    def calculate_velocity(current_positions, previous_positions, all_detections, previus_all_detections):
        """
        Calcula a velocidade de cada jogador.

        Args:
            current_positions (numpy.ndarray): Posições atuais dos jogadores [[x1, y1], ...].
            previous_positions (numpy.ndarray): Posições anteriores dos jogadores [[x1, y1], ...].
            all_detections: Objeto contendo IDs dos jogadores e suas posições.

        Returns:
            list: Uma lista de velocidades para cada jogador.
        """
        velocities = []
        for i in range(1, 11):  # IDs de 1 a 10
            # Filtrar as posições atuais e anteriores para o jogador com ID `i`
            current_idx = np.where(all_detections.tracker_id == i)[0]
            previous_idx = np.where(previus_all_detections.tracker_id == i)[0]
            
            if len(current_idx) == 0 or len(previous_idx) == 0:
                continue  # Pula se o jogador não foi detectado no frame atual ou anterior

            # Garantir que os índices não ultrapassam os limites
            if current_idx[0] >= len(current_positions) or previous_idx[0] >= len(previous_positions):
                continue

            current = current_positions[current_idx[0]]
            previous = previous_positions[previous_idx[0]]

            # Calcular a distância e a velocidade
            distance = np.sqrt(
                (current[0] / 100 - previous[0] / 100) ** 2 +
                (current[1] / 100 - previous[1] / 100) ** 2
            )
            velocities.append((i, distance * 3.6 / 0.34)) 

        return velocities


    @staticmethod
    def calculate_amplitude_and_depth(player_positions):
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
