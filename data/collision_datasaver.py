from pathlib import Path
import numpy as np
import io

class CollisionDataSaver:
	def __init__(self, folder, prefix, save_viz):
		self.n_saved = 0 
		self.prefix = prefix
		self.save_viz = save_viz
		self.folder = Path(folder)

		self.obs = []
		self.states = []
		self.collisions = []

		Path(self.folder).mkdir(parents=True, exist_ok=True)
		(self.folder / "simdata").mkdir(exist_ok=True)
		if self.save_viz:
			(self.folder / "viz").mkdir(exist_ok=True)

	def save(self, data):
		file_name = self.folder / "simdata" /  f"{self.prefix}_{self.n_saved}"

		with io.BytesIO() as bs:
			np.savez_compressed(bs, **data)
			bs.seek(0)
			with file_name.open('wb') as f:
				f.write(bs.read())
		self.n_saved += 1