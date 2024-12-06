from classes.footballgameprocessor import FootballGameProcessor

source_video_path = "test/test.mp4"
target_folder = "test/output"
processor = FootballGameProcessor(source_video_path, target_folder)
processor.run()