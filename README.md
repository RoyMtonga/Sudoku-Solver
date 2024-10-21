This is a program written in Python that connects with your webcam and tries to solve a popular puzzle called sudoku

## Code requirements
Python with following modules installed:
* NumPy
* TensorFlow
* Keras
* Matplotlib (to train a model that recognizes digits)
* OpenCV

## Usage
After running main_file.py you should see a window that shows live feed from your webcam.
Now place a sudoku in the webcam's field of view, a solution should appear in the window.
If the solution doesn't appear, or the program doesn't even locate the sudoku, try to move it closer/further to the webcam. If it doesn't help, you may need to improve the lighting quality.

## How does it work?
Short explanation - algorithm:
* read a frame from a webcam
* convert that frame into grayscale
* binarize that frame
* find all external contours
* get the biggest quadrangle from that contours
* apply warp transform (bird eye view) on the biggest quadrangle
* split that quadrangle into 81 small boxes
* check which boxes contain digits
* extract digits from boxes that aren't empty
* prepare that digits for a CNN model
* while not solved and iterations of the loop <= 4:
	* rotate the digits by (90 * current iteration) degrees
	* classify the digits using a CNN model
	* if an average probability is too low go to the next iteration of the loop
	* compare the digits with a previous solution
	* if the digits are part of the previous solution then we don't need to solve the sudoku again - break the loop
	* try to solve the sudoku
	* if solved correctly break the loop
* return a copy of the frame (with a solution if any were found)

Precise explanation - code analysis:  

The program starts in main_file.py.  
First of all we have to import the source code and some libraries from other files.
```python
print('Importing a source code and libraries from other files...')

from webcam_sudoku_solver import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # hide tf warnings
import tensorflow as tf
```

Then main function starts.  
First task of the function is to prepare a CNN model and a webcam.
```python
model = tf.keras.models.load_model('Models/handwritten_cnn.h5')

webcam_width, webcam_height = 1920, 1080
webcam = cv.VideoCapture(0)
webcam.set(cv.CAP_PROP_FRAME_WIDTH, webcam_width)
webcam.set(cv.CAP_PROP_FRAME_HEIGHT, webcam_height)
```

Now main loop of the program starts.  
We'll use there a object of WebcamSudokuSolver class - the core of the program.  
```python

webcam_sudoku_solver = WebcamSudokuSolver(model)
```

At the beginning of each iteration of main loop a frame is read from a webcam.  

Then that frame is passed as an argument to the object of WebcamSudokuSolver class using solve function.  
The function returns a copy of that frame (with a drawn solution if any has been found).  
How does solve function convert a webcam frame into a frame with solution?
But now let's see what happens with that returned frame.  
That frame is just displayed.
We also check if a user has pressed a key (if so, the program is closed).

```python
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
```

If there are no errors, the following information will be displayed at the very end of the program:  
"Code is done, so everything works fine!".  


First task of the function is to extract a sudoku board.


The program assumes that a sudoku board as the biggest quadrangle in a frame.
```python
if frame is None:
	return frame

frame = deepcopy(frame)

warp_sudoku_board, warp_matrix = get_biggest_quadrangle(frame)

if warp_sudoku_board is None:
	return frame
```
if the function won't solve the sudoku then it will return an unchaged copy of the frame.  

Next step is to split that board into 81 boxes.
```python
boxes = get_boxes(warp_sudoku_board)
```
Using trials and errors technique, I developed the following algorithm:
* copy a box
* crop that copy on each side by 15%
* find all external contours
* if there are no contours it means there is no digit - return False
* if there is at least one external contour get the biggest (only the biggest could be a digit)
* if an area of that contour is too small it means there is no digit - return False
* get a bounding rectangle of the biggest contour
* if width and height of that rectangle is too small it means there is no digit - return False
* return True - there is a digit

The algorithm is implemented in check_digits_occurrence function
```python
digits_occurrence = check_digits_occurrence(boxes)
```

Now it's time to get inputs for a CNN model from boxes that contain digits
```python
inputs = prepare_inputs(boxes, digits_occurrence)
if inputs is None:
	return frame
```

The program works with sudoku rotated in every way,  
but cropped and warped boards which are returned by get_biggest_quadrangle function may be rotated only in 4 ways - by 0, 90, 180 or 270 degrees.  
That's just how get_biggest_quadrangle function works.

We don't know which rotation is correct, so we need to try solve it even 4 times.
```python
current_attempt = 1
while current_attempt <= 4:
	rotation_angle = self.last_solved_sudoku_rotation + 90 * (current_attempt - 1)

	rotated_inputs = rotate_inputs(inputs, rotation_angle)

	predictions = self.model.predict([rotated_inputs])

	if not probabilities_are_good(predictions):
		current_attempt += 1
		continue

	digits_grid = get_digits_grid(predictions, digits_occurrence, rotation_angle)

	if self.new_sudoku_solution_may_be_last_solution(digits_grid):
		self.last_solved_sudoku_rotation = rotation_angle

		result = inverse_warp_digits_on_frame(
			digits_grid, self.last_sudoku_solution, frame, warp_sudoku_board.shape, warp_matrix, rotation_angle
		)

		return result

	solved_digits_grid = sudoku_solver.solve_sudoku(digits_grid)
	if solved_digits_grid is None:
		current_attempt += 1
		continue

	self.last_sudoku_solution = solved_digits_grid
	self.last_solved_sudoku_rotation = rotation_angle

	result = inverse_warp_digits_on_frame(
		digits_grid, solved_digits_grid, frame, warp_sudoku_board.shape, warp_matrix, rotation_angle
	)

	return result

return frame
```
this loop step by step.  

First an angle is calculated and inputs for a CNN model are rotated. 
```python
rotation_angle = self.last_solved_sudoku_rotation + 90 * (current_attempt - 1)

rotated_inputs = rotate_inputs(inputs, rotation_angle)
```

Now a CNN model can predict.
```python
predictions = self.model.predict([rotated_inputs])
```

If an average probability isn't high enough it means the current rotation isn't correct. We can skip to the next iteration.  
if it is 4th iteration then the function won't solve the sudoku and will return a copy of the frame without any changes. 
```python
if not probabilities_are_good(predictions):
	current_attempt += 1
	continue
```

If an average probability is high enough we can get a grid with recognized digits.
```python
digits_grid = get_digits_grid(predictions, digits_occurrence, rotation_angle)
```

This function always returns a "vertically normalized" grid so it always can be compared with the previous solution, regardless of their rotation.  

Comparing the current grid with the previous solution:
```python
if self.new_sudoku_solution_may_be_last_solution(digits_grid):
```

If a solution of the current grid can be equal to the previous solution we don't have to solve the current sudoku at all.
```python
if self.new_sudoku_solution_may_be_last_solution(digits_grid):
	self.last_solved_sudoku_rotation = rotation_angle

	result = inverse_warp_digits_on_frame(
		digits_grid, self.last_sudoku_solution, frame, warp_sudoku_board.shape, warp_matrix, rotation_angle
	)

	return result 
```

Otherwise solve function will try to solve the current sudoku.
```python
solved_digits_grid = sudoku_solver.solve_sudoku(digits_grid)
```

If that sudoku is unsolvable it means that the current rotation isn't correct after all.
```python
if solved_digits_grid is None:
	current_attempt += 1
	continue
```

But if that sudoku has been solved correctly we overwrite the previous solution.
```python
self.last_sudoku_solution = solved_digits_grid
self.last_solved_sudoku_rotation = rotation_angle
```

Draw the current solution on a copy of the current frame and return it.  


```python
result = inverse_warp_digits_on_frame(
	digits_grid, solved_digits_grid, frame, warp_sudoku_board.shape, warp_matrix, rotation_angle
)

return result
```

If we couldn't find any solution of the sudoku in any rotation, we return the image without any solution.
```python
return frame
```

How does solve_sudoku function solve a sudoku puzzle?
```python
solved_digits_grid = sudoku_solver.solve_sudoku(digits_grid)
```

First we need to check if the sudoku is solvable at all. 
```python
if not is_solvable(digits_grid):
	return None
```

The algorithm is based on pencilmarks that we use to help ourself solve sudoku in real life.  
```python
human_notes = get_full_human_notes(digits_grid)
```
The sudoku is solved in a loop.
```python
while True:
	sth_has_changed1 = remove_orphans_technique(digits_grid, human_notes)

	sth_has_changed2 = single_appearances_technique(digits_grid, human_notes)

	if not sth_has_changed1 and not sth_has_changed2:
		break
```
Each iteration of the loop calls two functions: remove_orphans_technique and single_appearances_technique.
Their task is to successively delete unnecessary notes and complete the sudoku.
The loop ends when the functions doesn't change anything anymore. It means the sudoku is solved or can't be solved using this technique.  
After the loop we check if that sudoku is solved correctly (so we check also if is solved at all).
```python
if is_solved_correctly(digits_grid):
	return digits_grid
return None
```



