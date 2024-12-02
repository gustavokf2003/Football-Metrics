from typing import Optional, List

import cv2
import supervision as sv
import numpy as np

from soccerpitchconfiguration import SoccerPitchConfiguration

class SoccerVisualizer:
    @staticmethod
    def draw_pitch(config, background_color=sv.Color(34, 139, 34), line_color=sv.Color.WHITE,
                    padding=50, line_thickness=4, point_radius=8, scale=0.1):
        """Desenha um campo de futebol com as dimensões e cores especificadas."""
        scaled_width = int(config.width * scale)
        scaled_length = int(config.length * scale)
        pitch_image = np.ones((scaled_width + 2 * padding, scaled_length + 2 * padding, 3),
                              dtype=np.uint8) * np.array(background_color.as_bgr(), dtype=np.uint8)
        for start, end in config.edges:
            p1 = (int(config.vertices[start - 1][0] * scale) + padding, int(config.vertices[start - 1][1] * scale) + padding)
            p2 = (int(config.vertices[end - 1][0] * scale) + padding, int(config.vertices[end - 1][1] * scale) + padding)
            cv2.line(img=pitch_image, pt1=p1, pt2=p2, color=line_color.as_bgr(), thickness=line_thickness)
        return pitch_image

    @staticmethod
    def draw_points_on_pitch(config, xy, face_color=sv.Color.RED, edge_color=sv.Color.BLACK,
                             radius=10, thickness=2, padding=50, scale=0.1, pitch=None):
        """Desenha pontos no campo de futebol."""
        if pitch is None:
            pitch = SoccerVisualizer.draw_pitch(config=config, padding=padding, scale=scale)
        for point in xy:
            scaled_point = (int(point[0] * scale) + padding, int(point[1] * scale) + padding)
            cv2.circle(img=pitch, center=scaled_point, radius=radius, color=face_color.as_bgr(), thickness=-1)
            cv2.circle(img=pitch, center=scaled_point, radius=radius, color=edge_color.as_bgr(), thickness=thickness)
        return pitch

    @staticmethod
    def draw_paths_on_pitch(config, paths, color=sv.Color.WHITE, thickness=2, padding=50, scale=0.1, pitch=None):
        """Desenha caminhos no campo de futebol."""
        if pitch is None:
            pitch = SoccerVisualizer.draw_pitch(config=config, padding=padding, scale=scale)
        for path in paths:
            scaled_path = [(int(point[0] * scale) + padding, int(point[1] * scale) + padding) for point in path if point.size > 0]
            if len(scaled_path) < 2:
                continue
            for i in range(len(scaled_path) - 1):
                cv2.line(img=pitch, pt1=scaled_path[i], pt2=scaled_path[i + 1], color=color.as_bgr(), thickness=thickness)
        return pitch

    @staticmethod
    def draw_pitch_voronoi_diagram(config, team_1_xy, team_2_xy, team_1_color=sv.Color.RED,
                                   team_2_color=sv.Color.WHITE, opacity=0.5, padding=50, scale=0.1, pitch=None):
        """Desenha um diagrama de Voronoi no campo representando áreas de controle."""
        if pitch is None:
            pitch = SoccerVisualizer.draw_pitch(config=config, padding=padding, scale=scale)
        scaled_width = int(config.width * scale)
        scaled_length = int(config.length * scale)
        voronoi = np.zeros_like(pitch, dtype=np.uint8)
        y_coords, x_coords = np.indices((scaled_width + 2 * padding, scaled_length + 2 * padding))
        y_coords -= padding
        x_coords -= padding

        def calc_distances(xy):
            return np.sqrt((xy[:, 0][:, None, None] * scale - x_coords) ** 2 + (xy[:, 1][:, None, None] * scale - y_coords) ** 2)

        dist_team_1 = calc_distances(team_1_xy)
        dist_team_2 = calc_distances(team_2_xy)
        control_mask = np.min(dist_team_1, axis=0) < np.min(dist_team_2, axis=0)

        voronoi[control_mask] = np.array(team_1_color.as_bgr(), dtype=np.uint8)
        voronoi[~control_mask] = np.array(team_2_color.as_bgr(), dtype=np.uint8)
        return cv2.addWeighted(voronoi, opacity, pitch, 1 - opacity, 0)

    @staticmethod
    def draw_text_on_frame(frame, team1_data, team2_data, posse1, posse2):
        # Configuração da fonte e estilo
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 1
        font_color1 = (0, 55, 255)  # Cor do texto para o time 1
        font_color2 = (63, 115, 75)  # Cor do texto para o time 2

        # Texto para o Time 1
        if team1_data[0] is not None:
            text_team1 = f"Team 1 - Amp: {team1_data[0] / 100:.2f}m, Depth: {team1_data[1] / 100:.2f}m"
        else:
            text_team1 = "Team 1 - No Data"

        if team2_data[0] is not None:
            text_team2 = f"Team 2 - Amp: {team2_data[0] / 100:.2f}m, Depth: {team2_data[1] / 100:.2f}m"
        else:
            text_team2 = "Team 2 - No Data"

        # Calculo da posse com verificação para evitar divisões por zero
        total_posse = posse1 + posse2
        if total_posse > 0:
            posse1_percent = posse1 / total_posse * 100
            posse2_percent = posse2 / total_posse * 100
        else:
            posse1_percent = 0
            posse2_percent = 0

        # Formatação do texto da posse
        text_team1_posse = f"Posse Time 1: {posse1_percent:.2f}%"
        text_team2_posse = f"Posse Time 2: {posse2_percent:.2f}%"

        # Posição inicial do texto
        height, width, _ = frame.shape
        position_team1 = (30, 20)
        position_team1_posse = (30, 40)
        position_team2 = (30, 60)
        position_team2_posse = (30, 80)

        # Adiciona o texto no frame
        cv2.putText(frame, text_team1, position_team1, font, font_scale, font_color1, thickness, cv2.LINE_AA)
        cv2.putText(frame, text_team1_posse, position_team1_posse, font, font_scale, font_color1, thickness, cv2.LINE_AA)
        cv2.putText(frame, text_team2, position_team2, font, font_scale, font_color2, thickness, cv2.LINE_AA)
        cv2.putText(frame, text_team2_posse, position_team2_posse, font, font_scale, font_color2, thickness, cv2.LINE_AA)

