from typing import Optional, List, Tuple, Dict

import cv2
import supervision as sv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mplsoccer import Pitch
from mplsoccer import heatmap
from scipy.ndimage import gaussian_filter
import os


from classes.soccerpitchconfiguration import SoccerPitchConfiguration


class SoccerVisualizer:
    @staticmethod
    def draw_pitch(
        config: SoccerPitchConfiguration,
        background_color: sv.Color = sv.Color(34, 139, 34),
        line_color: sv.Color = sv.Color.WHITE,
        padding: int = 50,
        line_thickness: int = 4,
        scale: float = 0.1
    ) -> np.ndarray:
        """
        Draws a football field with the specified dimensions and colors.

        Args:
            config (object): Object containing the field's dimensions and vertices.
            background_color (str): Background color of the field.
            line_color (str): Color of the field lines.
            padding (int): Space around the field.
            line_thickness (int): Thickness of the field lines.
            scale (float): Scaling factor to adjust the field dimensions.

        Returns:
            Image: Image of the drawn football field.
        """

        # Ajusta as dimensões do campo de acordo com a escala
        scaled_width = int(config.width * scale)
        scaled_length = int(config.length * scale)

        # Cria uma imagem em branco (campo) com as dimensões escaladas e cor de fundo
        pitch_image = np.ones((scaled_width + 2 * padding, scaled_length + 2 * padding, 3),
                              dtype=np.uint8) * np.array(background_color.as_bgr(), dtype=np.uint8)

        # Desenha as linhas do campo com as coordenadas dos vértices
        for start, end in config.edges:
            p1 = (int(config.vertices[start - 1][0] * scale) + padding, int(config.vertices[start - 1][1] * scale) + padding)
            p2 = (int(config.vertices[end - 1][0] * scale) + padding, int(config.vertices[end - 1][1] * scale) + padding)
            cv2.line(img=pitch_image, pt1=p1, pt2=p2, color=line_color.as_bgr(), thickness=line_thickness)

        return pitch_image

    @staticmethod
    def draw_points_on_pitch(
        config: SoccerPitchConfiguration,
        xy: List[Tuple[float, float]],
        face_color: sv.Color = sv.Color.RED,
        edge_color: sv.Color = sv.Color.BLACK,
        radius: int = 10,
        thickness: int = 2,
        padding: int = 50,
        scale: float = 0.1,
        pitch: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Draws points on the football field.

        Args:
            config (object): Object containing the field's dimensions and vertices.
            xy (list): List of coordinates (x, y) to draw the points.
            face_color (str): Fill color of the points.
            edge_color (str): Border color of the points.
            radius (int): Radius of the points.
            thickness (int): Border thickness of the points.
            padding (int): Space around the field.
            scale (float): Scaling factor to adjust the field dimensions.
            pitch (Image): Image of the field where the points will be drawn.

        Returns:
            Image: Image of the field with the points drawn.
        """

        # Se a imagem do campo não for fornecida, desenha o campo primeiro
        if pitch is None:
            pitch = SoccerVisualizer.draw_pitch(config=config, padding=padding, scale=scale)

        # Desenha cada ponto na imagem
        for point in xy:
            # Escala as coordenadas do ponto e adiciona o padding
            scaled_point = (int(point[0] * scale) + padding, int(point[1] * scale) + padding)

            # Desenha o ponto com a cor de preenchimento e borda
            cv2.circle(img=pitch, center=scaled_point, radius=radius, color=face_color.as_bgr(), thickness=-1)
            cv2.circle(img=pitch, center=scaled_point, radius=radius, color=edge_color.as_bgr(), thickness=thickness)

        return pitch

    @staticmethod
    def draw_paths_on_pitch(
        config: SoccerPitchConfiguration,
        paths: List[List[Tuple[float, float]]],
        color: sv.Color = sv.Color.WHITE,
        thickness: int = 2,
        padding: int = 50,
        scale: float = 0.1,
        pitch: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Draws paths on the football field.

        Args:
            config (object): Object containing the field's dimensions and vertices.
            paths (list): List of paths (lists of coordinates) to be drawn.
            color (str): Color of the path lines.
            thickness (int): Thickness of the path lines.
            padding (int): Space around the field.
            scale (float): Scaling factor to adjust the field dimensions.
            pitch (Image): Image of the field where the paths will be drawn.

        Returns:
            Image: Image of the field with the paths drawn.
        """

        # Se a imagem do campo não for fornecida, desenha o campo primeiro
        if pitch is None:
            pitch = SoccerVisualizer.draw_pitch(config=config, padding=padding, scale=scale)

        # Desenha cada caminho na imagem
        for path in paths:
            # Escala cada ponto do caminho e adiciona o padding
            scaled_path = [(int(point[0] * scale) + padding, int(point[1] * scale) + padding) for point in path if point]

            # Ignora caminhos com menos de 2 pontos
            if len(scaled_path) < 2:
                continue

            # Desenha as linhas conectando os pontos do caminho
            for i in range(len(scaled_path) - 1):
                cv2.line(img=pitch, pt1=scaled_path[i], pt2=scaled_path[i + 1], color=color.as_bgr(), thickness=thickness)

        return pitch

    @staticmethod
    def draw_pitch_voronoi_diagram(
        config: SoccerPitchConfiguration,
        team_1_xy: List[Tuple[float, float]],
        team_2_xy: List[Tuple[float, float]],
        team_1_color: sv.Color = sv.Color.RED,
        team_2_color: sv.Color = sv.Color.WHITE,
        opacity: float = 0.5,
        padding: int = 50,
        scale: float = 0.1,
        pitch: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Draws a Voronoi diagram on the field representing control areas.

        Args:
            config (object): Object containing the field's dimensions and vertices.
            team_1_xy (list): Coordinates of team 1's players.
            team_2_xy (list): Coordinates of team 2's players.
            team_1_color (str): Color for team 1's area.
            team_2_color (str): Color for team 2's area.
            opacity (float): Opacity of the overlay between the diagram and the field.
            padding (int): Space around the field.
            scale (float): Scaling factor to adjust the field dimensions.
            pitch (Image): Image of the field where the diagram will be drawn.

        Returns:
            Image: Image of the field with the Voronoi diagram drawn.
        """
        
        # Se a imagem do campo não for fornecida, desenha o campo primeiro
        if pitch is None:
            pitch = SoccerVisualizer.draw_pitch(config=config, padding=padding, scale=scale)

        # Obtém as dimensões escaladas do campo
        scaled_width = int(config.width * scale)
        scaled_length = int(config.length * scale)

        # Cria uma imagem preta para o diagrama de Voronoi
        voronoi = np.zeros_like(pitch, dtype=np.uint8)
        y_coords, x_coords = np.indices((scaled_width + 2 * padding, scaled_length + 2 * padding))
        y_coords -= padding
        x_coords -= padding

        # Calcula as distâncias de cada ponto (jogador) ao campo
        def calc_distances(xy: np.ndarray) -> np.ndarray:
            return np.sqrt((xy[:, 0][:, None, None] * scale - x_coords) ** 2 + (xy[:, 1][:, None, None] * scale - y_coords) ** 2)

        dist_team_1 = calc_distances(np.array(team_1_xy))
        dist_team_2 = calc_distances(np.array(team_2_xy))

        # Cria uma máscara para determinar as áreas de controle de cada time
        control_mask = np.min(dist_team_1, axis=0) < np.min(dist_team_2, axis=0)

        # Aplica as cores dos times nas áreas de controle
        voronoi[control_mask] = np.array(team_1_color.as_bgr(), dtype=np.uint8)
        voronoi[~control_mask] = np.array(team_2_color.as_bgr(), dtype=np.uint8)

        # Sobrepõe o diagrama de Voronoi com a opacidade definida
        return cv2.addWeighted(pitch, 1 - opacity, voronoi, opacity, 0)
    
    @staticmethod
    def draw_text_on_frame(
        frame: np.ndarray,
        team1_data: Tuple[float, float],
        team2_data: Tuple[float, float],
        posse1: int, 
        posse2: int):

        """
        Draws text on the frame displaying data about the teams and possession.

        Args:
            frame (ndarray): The image/frame where the text will be drawn.
            team1_data (tuple): A tuple containing the amplitude and depth data for team 1.
            team2_data (tuple): A tuple containing the amplitude and depth data for team 2.
            posse1 (int): Number of frames team 1 has possession.
            posse2 (int): Number of frames team 2 has possession. 

        Returns:
            None: The function modifies the input frame by adding text directly onto it.
        """

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


    def plot_gaussian_heatmap(player_positions, target_folder, field_size=(3000, 4600), sigma=1):
        """
        Creates a Gaussian-smoothed heatmap for each player based on their positions.

        Args:
            player_positions (dict): Dictionary where the key is the player ID, and the value is a list of arrays
                                     representing the player's positions on the field [[x1, y1], [x2, y2], ...].
            field_size (tuple): Field dimensions (width, height), default is 3000x4500.
            grid_size (int): Grid size for the histogram resolution.
            sigma (float): Parameter for Gaussian smoothing.

        Returns:
            None
        """
        pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#efefef')

        for player_id, positions in player_positions.items():
            # Extrair coordenadas X e Y
            x_data, y_data = [], []
            for position in positions:
                if position.size > 0:  # Ignorar arrays vazios
                    x_data.append(position[0, 0])
                    y_data.append(position[0, 1])
            
            x_positions = np.array(x_data)
            y_positions = np.array(y_data)

            # Normalizar as posições
            x_scale = 68 / field_size[0]  # Largura do campo
            y_scale = 105 / field_size[1]  # Altura do campo
            x_positions = x_positions * x_scale
            y_positions = y_positions * y_scale

            # Configuração do gráfico
            fig, ax = pitch.draw(figsize=(6.6, 4.125))
            fig.set_facecolor('#22312b')

            # Criar o histograma 2D
            bin_statistic = pitch.bin_statistic(
                y_positions, x_positions, 
                statistic='count', 
                bins=(25, 25)
            )
            
            # Aplicar suavização Gaussiana
            bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], sigma=sigma)

            # Mapa de calor
            pcm = pitch.heatmap(bin_statistic, ax=ax, cmap='hot', edgecolors='#22312b', alpha=0.75)

            # Barra de cor
            cbar = fig.colorbar(pcm, ax=ax, shrink=0.6)
            cbar.outline.set_edgecolor('#efefef')
            cbar.ax.yaxis.set_tick_params(color='#efefef')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#efefef')

            # Exibir o título
            ax.set_title(f"Mapa de Calor - Jogador {player_id}", color='white', fontsize=14)
            filepath = os.path.join(target_folder, f"heatmap_player_{player_id}.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)  # Fecha o gráfico para liberar memória

