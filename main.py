# This is where this program starts and ends.
# To exit the program press any key.

print('Importing code and libraries from other files...')

from webcam_sudoku_solver import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # hide tf warnings
import tensorflow as tf


def main():
	model = tf.keras.models.load_model('models/handwritten_cnn.h5')

	webcam_width, webcam_height = 1920, 1080
	webcam = cv.VideoCapture(0)
	webcam.set(cv.CAP_PROP_FRAME_WIDTH, webcam_width)
	webcam.set(cv.CAP_PROP_FRAME_HEIGHT, webcam_height)

	# create the core of the program
	webcam_sudoku_solver = WebcamSudokuSolver(model)

	print('Logs:')
	while webcam.isOpened():
		successful_frame_read, frame = webcam.read()

		if not successful_frame_read:
			break

		# run the core of the program
		output_frame = webcam_sudoku_solver.solve(frame)

		# output results
		cv.imshow('Webcam Sudoku Solver', output_frame)

		# check if a user has pressed a key, if so, close the program
		if cv.waitKey(1) >= 0:
			break

	cv.destroyAllWindows()
	webcam.release()


if __name__ == "__main__":
	main()

print('Code is done, so everything works fine!')
