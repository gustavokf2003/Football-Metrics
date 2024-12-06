import supervision as sv
from sports.common.view import ViewTransformer
from sports.common.team import TeamClassifier
import cv2
import numpy as np
from tqdm import tqdm
import copy
from classes.soccerpitchconfiguration import SoccerPitchConfiguration
from classes.calculator import Calculator
from classes.soccervisualizer import SoccerVisualizer
from inference import get_model
import os

class FootballGameProcessor:
    def __init__(self, source_video_path, target_folder):
        # Verifica se a pasta existe
        if not os.path.exists(target_folder):
            # Se não existir, cria a pasta
            os.makedirs(target_folder)

        self.target_folder = target_folder
        self.metrics_output = os.path.join(target_folder, "metrics.txt")

        # Caminho para o vídeo de entrada e saída
        self.source_video_path = source_video_path
        self.target_video_path = os.path.join(target_folder, "output_video.mp4")

        # Carrega informações sobre o vídeo
        self.video_info = sv.VideoInfo.from_video_path(self.source_video_path)
        self.video_sink = sv.VideoSink(self.target_video_path, self.video_info)

        # Gera um iterador para os frames do vídeo
        self.frame_generator = sv.get_video_frames_generator(self.source_video_path)

        # IDs das classes para diferentes objetos
        self.BALL_ID = 0
        self.GOALKEEPER_ID = 1
        self.PLAYER_ID = 2

        # Inicializa anotadores para diferentes tipos de anotações
        self.color_palette = sv.ColorPalette.from_hex(['#ff3700', '#4b733f', '#FFD700'])
        self.ellipse_annotator = sv.EllipseAnnotator(color=self.color_palette, thickness=2)
        self.label_annotator = sv.LabelAnnotator(color=self.color_palette, text_color=sv.Color.from_hex('#000000'),
                                                  text_position=sv.Position.BOTTOM_CENTER, text_scale=0.3)
        self.triangle_annotator = sv.TriangleAnnotator(color=sv.Color.from_hex('#FFD700'), base=20, height=17)

        # Inicializa o rastreador ByteTrack para acompanhar os objetos detectados
        self.tracker = sv.ByteTrack(
            track_thresh=0.2,
            track_buffer=60,
            match_thresh=0.9,
            frame_rate=30
        )
        self.tracker.reset()

        # Carrega o modelo de detecção de jogadores 
        self.player_detection_model = get_model(model_id="football-players-detection-mwgyr/11", api_key='MvwAvVqqfRPNGbScb4oP')

        # Carrega o modelo de detecção de pontos no campo 
        self.field_detection_model = get_model(model_id="keypoint-football-field-detect/5", api_key='MvwAvVqqfRPNGbScb4oP')

        # Inicializa o modelo para classificação de times
        self.team_classifier = TeamClassifier(device="cuda")
        self.train_team_classifier()

        # Configuração do campo
        self.config = SoccerPitchConfiguration()

        # Inicializa dicionário para as posições dos jogadores
        self.positions = {id: None for id in range(1, 11)}
        self.passes = {id: 0 for id in range(1, 11)}
        self.player_positions = {id: [] for id in range(1, 11)}
        self.player_velocities = {id: [0] for id in range(1, 11)}
        self.team_players = {team: set() for team in range(1, 3)}
        self.max_id_assigned = 10
        self.last_player_with_ball = None
        self.positions_early = None

        # Métricas para os times
        self.metrics = {
            1: {
                'Centroide': [],
                'Amplitude': [],
                'Profundidade': [],
                'Posse': 0,
                'Passes': 0
            },
            2: {
                'Centroide': [],
                'Amplitude': [],
                'Profundidade': [],
                'Posse': 0,
                "Passes": 0
            }
        }

    def resolve_goalkeepers_team_id(self, players: sv.Detections, goalkeepers: sv.Detections) -> np.ndarray:
        # Obtém as coordenadas dos goleiros e dos jogadores
        goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        
        # Calcula o centróide de cada time (time 0 e time 1)
        team_0_centroid = players_xy[players.class_id == 0].mean(axis=0)
        team_1_centroid = players_xy[players.class_id == 1].mean(axis=0)
        
        # Lista para armazenar o time de cada goleiro
        goalkeepers_team_id = []
        
        # Calcula a distância de cada goleiro aos times e atribui o time mais próximo
        for goalkeeper_xy in goalkeepers_xy:
            dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
            dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
            goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
        
        return np.array(goalkeepers_team_id)

    def train_team_classifier(self):

        # Gera um iterador para os frames do vídeo, processando um frame a cada 30 frames
        frame_generator = sv.get_video_frames_generator(
            source_path=self.source_video_path, stride=30)

        # Inicializa uma lista vazia para armazenar as partes recortadas (crops) das imagens dos jogadores
        crops = []

        # Itera sobre cada frame gerado, exibindo uma barra de progresso com a descrição 'collecting crops'
        for frame in tqdm(frame_generator, desc='collecting crops'):
            # Executa a inferência do modelo de detecção de jogadores no frame com confiança mínima de 0.3
            result = self.player_detection_model.infer(frame, confidence=0.4)[0]

            # Converte os resultados da inferência em um objeto de detecções utilizável
            detections = sv.Detections.from_inference(result)

            # Filtra as detecções para manter apenas aquelas relacionadas ao jogador com o ID especificado
            players_detections = detections[detections.class_id == self.PLAYER_ID]

            # Recorta as áreas da imagem onde os jogadores foram detectados e armazena em `players_crops`
            players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]

            # Adiciona os recortes (crops) dos jogadores à lista `crops`
            crops += players_crops

        # Treina o classificador com as imagens recortadas dos jogadores
        self.team_classifier.fit(crops)

    def process_detection(self, frame):
        # Realiza a detecção de objetos no frame atual
        result = self.player_detection_model.infer(frame, confidence=0.4)[0]
        detections = sv.Detections.from_inference(result)

        # Separa as detecções de bola, goleiros, jogadores e árbitros
        ball_detections = detections[detections.class_id == self.BALL_ID]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

        # Filtra e atualiza as detecções de jogadores e árbitros
        all_detections = detections[detections.class_id != self.BALL_ID]
        all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)

        return all_detections, ball_detections
    
    def track_players(self, all_detections):

        # Atualiza as detecções com o rastreador
        all_detections = self.tracker.update_with_detections(detections=all_detections)

        # Verifica se há IDs faltando
        current_ids = set(all_detections.tracker_id)
        missing_ids = set(self.positions.keys()) - current_ids

        # Prepara listas para novas detecções com IDs atualizados
        updated_detections = { "xyxy": [], "mask": [], "confidence": [], "class_id": [], "tracker_id": [], "data": [] }
                
        for detection in all_detections:
            tracker_id_d = detection[4]  # Acessa o tracker_id atual

            # Verifica se o ID é maior que o max_id_assigned e há IDs faltando
            if tracker_id_d > self.max_id_assigned and missing_ids:
                # Verifica a posição mais próxima com os missing_ids
                distances = [np.linalg.norm(self.positions[mid] - detection[0]) for mid in missing_ids if self.positions[mid] is not None]
                closest_id = list(missing_ids)[np.argmin(distances)]
                updated_detections["tracker_id"].append(closest_id)
            else:
                updated_detections["tracker_id"].append(detection[4])

            # Preenche as listas de detecções
            updated_detections["xyxy"].append(detection[0])
            updated_detections["mask"].append(detection[1])
            updated_detections["confidence"].append(detection[2])
            updated_detections["class_id"].append(detection[3])
            updated_detections["data"].append(detection[5])

        # Cria um novo objeto de detecções com os novos IDs
        new_detections = sv.Detections(
            np.array(updated_detections["xyxy"]),
            None,
            np.array(updated_detections["confidence"]),
            np.array(updated_detections["class_id"]),
            np.array(updated_detections["tracker_id"]),
            data={"class_name": np.array(updated_detections["data"])}
        )
                
        # Atualiza as detecções com o rastreador
        all_detections = self.tracker.update_with_detections(detections=new_detections)

        return all_detections
    
    def separate_players(self, frame, all_detections):
        # Separa as detecções de goleiros e jogadores
        goalkeepers_detections = all_detections[all_detections.class_id == self.GOALKEEPER_ID]
        players_detections = all_detections[all_detections.class_id == self.PLAYER_ID]

        # Atribui times aos jogadores
        players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
        players_detections.class_id = self.team_classifier.predict(players_crops)

        # Resolve a atribuição de times para goleiros
        goalkeepers_detections.class_id = self.resolve_goalkeepers_team_id(players_detections, goalkeepers_detections)

        # Mescla todas as detecções em uma única coleção
        all_detections = sv.Detections.merge([players_detections, goalkeepers_detections])

        # Converte as IDs das classes para inteiros
        all_detections.class_id = all_detections.class_id.astype(int)

        return all_detections
    
    def transform_coordenates(self, frame, all_detections, ball_detections):
        # Detecta pontos de referência no campo
        result = self.field_detection_model.infer(frame, confidence=0.3)[0]
        key_points = sv.KeyPoints.from_inference(result)
                
        # Filtra pontos de referência
        filter = key_points.confidence[0] > 0.5
        frame_reference_points = key_points.xy[0][filter]
        pitch_reference_points = np.array(self.config.vertices)[filter]

        # Cria um transformador de visão para mapear pontos do frame para o campo
        transformer = ViewTransformer(source=frame_reference_points, target=pitch_reference_points)

        # Transforma as coordenadas da bola e dos jogadores para o campo
        frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

        players_xy = all_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pitch_players_xy = transformer.transform_points(points=players_xy)

        return pitch_ball_xy, pitch_players_xy
    
    def annotate_map2d(self, frame1, frame2, x_offset_scalar):
        # Ajusta a rotação e o posicionamento da imagem
        angle = 90
        (h, w) = frame2.shape[:2]
        center = (w // 2, h // 2)
        cos = np.abs(np.cos(np.radians(angle)))
        sin = np.abs(np.sin(np.radians(angle)))
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]
        rotated_frame = cv2.warpAffine(frame2, rotation_matrix, (new_w, new_h))
        flipped_frame = cv2.flip(rotated_frame, 0)

        # Ajusta o tamanho de 'frame2' para que caiba no canto inferior de 'frame1'
        h1, w1 = frame1.shape[:2]  # Dimensões da imagem principal
        h2, w2 = flipped_frame.shape[:2]  # Dimensões da imagem a ser inserida

        # Redimensiona 'frame2' para ser menor, se necessário
        scale_factor = 0.5  # Reduz o tamanho para 25% do original, ajuste conforme necessário
        new_w = int(w2 * scale_factor)
        new_h = int(h2 * scale_factor)
        resized_frame2 = cv2.resize(flipped_frame, (new_w, new_h))

        # Define a posição no canto inferior de 'frame1' onde a imagem será inserida
        x_offset = x_offset_scalar * (w1 - new_w) // 4 
        y_offset = h1 - new_h

        # Recorte a região do frame onde a imagem será sobreposta
        roi = frame1[y_offset:y_offset + new_h, x_offset:x_offset + new_w]

        # Ajuste o nível de transparência (0.0 = totalmente transparente, 1.0 = totalmente opaco)
        alpha = 0.7  # Transparência da imagem sobreposta
        beta = 1 - alpha  # Transparência do frame de fundo

        # Mistura a imagem com a região de interesse (ROI) do frame
        blended = cv2.addWeighted(resized_frame2, alpha, roi, beta, 0)

        # Insere a imagem misturada no canto inferior de 'frame1'
        frame1[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = blended

        return frame1
    
    def annotate_frame(self, frame, all_detections, pitch_ball_xy, pitch_players_xy, team1_centroid, team2_centroid):
        """
        Visualiza o campo com uma visão estilo radar de vídeo e desenha informações no frame.
        """

        # 1. Visualização inicial do campo
        map2d = SoccerVisualizer.draw_pitch(self.config)

        # 2. Desenho dos jogadores do time 1
        map2d = SoccerVisualizer.draw_points_on_pitch(
            config=self.config,
            xy=pitch_players_xy[all_detections.class_id == 0],
            face_color=sv.Color.from_hex('ff3700'),
            edge_color=sv.Color.BLACK,
            radius=16,
            pitch=map2d
        )

        # 3. Desenho dos jogadores do time 2
        map2d = SoccerVisualizer.draw_points_on_pitch(
            config=self.config,
            xy=pitch_players_xy[all_detections.class_id == 1],
            face_color=sv.Color.from_hex('4b733f'),
            edge_color=sv.Color.BLACK,
            radius=16,
            pitch=map2d
        )

        # 4. Desenho da bola
        map2d = SoccerVisualizer.draw_points_on_pitch(
            config=self.config,
            xy=pitch_ball_xy,
            face_color=sv.Color.WHITE,
            edge_color=sv.Color.BLACK,
            radius=10,
            pitch=map2d
        )

        # 5. Desenho dos centroides dos times
        map2d = SoccerVisualizer.draw_points_on_pitch(
            config=self.config,
            xy=team1_centroid,
            face_color=sv.Color.from_hex('fca790'),
            edge_color=sv.Color.BLACK,
            radius=10,
            pitch=map2d
        )
        map2d = SoccerVisualizer.draw_points_on_pitch(
            config=self.config,
            xy=team2_centroid,
            face_color=sv.Color.from_hex('a1fa87'),
            edge_color=sv.Color.BLACK,
            radius=10,
            pitch=map2d
        )

        # 6. Visualização Voronoi
        voronoi = SoccerVisualizer.draw_pitch(
            config=self.config,
            background_color=sv.Color.WHITE,
            line_color=sv.Color.BLACK
        )
        voronoi = SoccerVisualizer.draw_pitch_voronoi_diagram(
            config=self.config,
            team_1_xy=pitch_players_xy[all_detections.class_id == 0],
            team_2_xy=pitch_players_xy[all_detections.class_id == 1],
            team_1_color=sv.Color.from_hex('ff3700'),
            team_2_color=sv.Color.from_hex('4b733f'),
            pitch=voronoi
        )

        # 7. Adição da bola e jogadores à Voronoi
        voronoi = SoccerVisualizer.draw_points_on_pitch(
            config=self.config,
            xy=pitch_ball_xy,
            face_color=sv.Color.WHITE,
            edge_color=sv.Color.WHITE,
            radius=8,
            thickness=1,
            pitch=voronoi
        )
        voronoi = SoccerVisualizer.draw_points_on_pitch(
            config=self.config,
            xy=pitch_players_xy[all_detections.class_id == 0],
            face_color=sv.Color.from_hex('ff3700'),
            edge_color=sv.Color.WHITE,
            radius=16,
            thickness=1,
            pitch=voronoi
        )
        voronoi = SoccerVisualizer.draw_points_on_pitch(
            config=self.config,
            xy=pitch_players_xy[all_detections.class_id == 1],
            face_color=sv.Color.from_hex('4b733f'),
            edge_color=sv.Color.WHITE,
            radius=16,
            thickness=1,
            pitch=voronoi
        )

        # 8. Anotação do frame
        annotated_frame = self.annotate_map2d(frame, voronoi, 3)
        annotated_frame = self.annotate_map2d(annotated_frame, map2d, 1)

        return annotated_frame


    def run(self):
        counter = 0

        with self.video_sink:
            # Itera sobre cada frame do vídeo, exibindo uma barra de progresso
            for frame in tqdm(self.frame_generator, total=self.video_info.total_frames):
                
                # Realiza a detecção de objetos no frame atual
                all_detections, ball_detections = self.process_detection(frame)

                # Atribui IDs aos jogadores e goleiros
                all_detections = self.track_players(all_detections)

                # Separa os jogadores em times
                all_detections = self.separate_players(frame, all_detections)

                 # Atualiza as posições anteriores
                for detection in all_detections:
                    if detection[3] == 0 and detection[4] not in self.team_players[2]:
                        self.team_players[1].add(detection[4])
                    elif detection[3] == 1 and detection[4] not in self.team_players[1]:
                        self.team_players[2].add(detection[4])
                    self.positions[detection[4]] = detection[0]  # Atualiza a posição do tracker_id

                # Detecta pontos de referência no campo
                pitch_ball_xy, pitch_players_xy = self.transform_coordenates(frame, all_detections, ball_detections)

                player_with_ball, distance = Calculator.nearest_player_to_ball(pitch_ball_xy, pitch_players_xy, all_detections)

                if player_with_ball != None and self.last_player_with_ball != None and player_with_ball != self.last_player_with_ball:
                    if all_detections[all_detections.tracker_id == player_with_ball].class_id == all_detections[all_detections.tracker_id == self.last_player_with_ball].class_id:
                        self.passes[self.last_player_with_ball] += 1
                        self.metrics[1 if self.last_player_with_ball in self.team_players[1] else 2]['Passes'] += 1
                        self.last_player_with_ball = player_with_ball

                if player_with_ball != None and distance < 150:
                    team = all_detections[all_detections.tracker_id == player_with_ball].class_id[0] + 1
                    self.metrics[team]['Posse'] += 1
                    self.last_player_with_ball = player_with_ball

                # Calcular o centroide de cada time
                team1_centroid = Calculator.calculate_centroid(pitch_players_xy[all_detections.class_id == 0])
                team2_centroid = Calculator.calculate_centroid(pitch_players_xy[all_detections.class_id == 1])
                
                # Calcular a amplitude e profundidade de cada time
                team1_amplitude_depth = Calculator.calculate_amplitude_and_depth(pitch_players_xy[all_detections.class_id == 0])
                team2_amplitude_depth = Calculator.calculate_amplitude_and_depth(pitch_players_xy[all_detections.class_id == 1])

                # Anota o frame com elipses, rótulos e triângulos
                annotated_frame = frame.copy()

                # Anota o frame com informações visuais
                annotated_frame = self.annotate_frame(annotated_frame, all_detections, pitch_ball_xy, pitch_players_xy, team1_centroid, team2_centroid)

                # Anota o frame com texto
                SoccerVisualizer.draw_text_on_frame(annotated_frame, (team1_amplitude_depth[0][0], team1_amplitude_depth[0][1]), 
                        (team2_amplitude_depth[0][0], team2_amplitude_depth[0][1]), self.metrics[1]['Posse'], self.metrics[2]['Posse'])

                counter += 1
                if counter % 10 == 0:
                    if counter == 30:
                        positions_early = copy.deepcopy(pitch_players_xy)
                        all_detections_early = copy.deepcopy(all_detections)

                    self.metrics[1]['Centroide'].append(team1_centroid)
                    self.metrics[1]['Amplitude'].append(team1_amplitude_depth[0][0])
                    self.metrics[1]['Profundidade'].append(team1_amplitude_depth[0][1])
                    
                    self.metrics[2]['Centroide'].append(team1_centroid)
                    self.metrics[2]['Amplitude'].append(team2_amplitude_depth[0][0])
                    self.metrics[2]['Profundidade'].append(team2_amplitude_depth[0][1])

                    for i in range(1, 11):
                        self.player_positions[i].append(pitch_players_xy[all_detections.tracker_id == i])

                    if counter % 30 == 0:
                        velocidades = Calculator.calculate_velocity(pitch_players_xy, positions_early, all_detections, all_detections_early)
                        for i, speed in velocidades.items():
                            self.player_velocities[i].append(speed)

                    positions_early = copy.deepcopy(pitch_players_xy)
                    all_detections_early = copy.deepcopy(all_detections)
                                      
                labels = [f"#{tracker_id} - {self.player_velocities[tracker_id][-1]:.2f} km/h - {self.passes[tracker_id]} passes" for tracker_id in all_detections.tracker_id]
                annotated_frame = self.ellipse_annotator.annotate(scene=annotated_frame, detections=all_detections)
                annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=all_detections, labels=labels)
                annotated_frame = self.triangle_annotator.annotate(scene=annotated_frame, detections=ball_detections)

                self.video_sink.write_frame(annotated_frame)

        # Nome do arquivo de saída
        metrics = Calculator.calculate_average_metrics(self.metrics)
        speed = Calculator.calculate_average_speeds(self.player_velocities, self.team_players)
        entropy = Calculator.calculate_shannon_entropy_for_players(self.player_positions)
        SoccerVisualizer.plot_gaussian_heatmap(self.player_positions, self.target_folder)

        # Gerar o arquivo
        with open(self.metrics_output, "w") as arquivo:
            arquivo.write("Metricas Finais do Jogo\n")
            for team, metric in metrics.items():
                arquivo.write("=" * 30 + "\n\n")
                arquivo.write(f"Time {team}\n")
                for metric_name, value in metric.items():
                    arquivo.write(f"{metric_name}: [{value}]\n")

                arquivo.write("\n")
                arquivo.write(f"Passes total do time: {self.metrics[team]['Passes']}\n")
                arquivo.write("\n")
                arquivo.write("Passes por jogador:\n")
                for tracker_id, passes in self.passes.items():
                    if self.team_players[team] and tracker_id in self.team_players[team]:
                        arquivo.write(f"Jogador {tracker_id}: {passes}\n")

                arquivo.write("\n")
                arquivo.write(f"Velocidade media da equipe: {speed['media_equipe'][team]:.2f} km/h\n")
                arquivo.write("\n")
                arquivo.write("Velocidade media por jogador:\n")
                for tracker_id, velocidade in speed['media_individual'].items():
                    if velocidade[0] == team:
                        arquivo.write(f"Jogador {tracker_id}: {velocidade[1]:.2f} km/h\n")

                arquivo.write("\n")
                arquivo.write("Entropia de Shannon por jogador:\n")
                for tracker_id, entropia in entropy.items():
                    if tracker_id in self.team_players[team]:
                        arquivo.write(f"Jogador {tracker_id}: {entropia:.2f}\n")