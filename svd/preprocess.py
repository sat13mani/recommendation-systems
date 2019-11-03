import numpy as np


def preprocess(input_matrix):
	"""
	Preprocesses the input matrix to replace the NA values logically so that SVD can be performed.
	Also introduces biases to input matrix
	:param input_matrix: User-Movie rating matrix where matrix[i][j] represents rating given by
	user i for movie j
	:return preprocessed matrix which can be used for calculating SVD
	"""
	matrix = np.asarray(input_matrix, dtype=np.float32)
	mean = matrix.mean()

	# Calculate biases for users and items
	row_count, col_count = [], []
	for x in range(len(input_matrix)):
		row_count.append(np.count_nonzero(matrix[x,:]))
	for x in range(len(matrix[0])):
		col_count.append(np.count_nonzero(matrix[:,x]))

	row_means, col_means = [], []
	for x in range(len(matrix)):
		row_means.append((np.sum(matrix[x,:]) - (mean*row_count[x])) / (row_count[x] * row_count[x]))
	for x in range(len(matrix[0])):
		col_means.append((np.sum(matrix[:,x]) - (mean*col_count[x])) / (col_count[x] * col_count[x]))

	# Replace NA values so that matrix can be used for calculating SVD
	for x in range(len(matrix)):
		for y in range(len(matrix[0])):
			if matrix[x][y] == 0:
				matrix[x][y] = mean + row_means[x] + col_means[y]

			if matrix[x][y] > 5:
				matrix[x][y] = 5
			
			if matrix[x][y] < 1:
				matrix[x][y] = 1

	return matrix


if __name__ == '__main__':
	input_matrix = [[1, 1, 1, 0, 0],
					[3, 3, 3, 0, 0],
					[4, 4, 4, 0, 0],
					[5, 5, 5, 0, 0],
					[0, 2, 0, 4, 4],
					[0, 0, 0, 5, 5],
					[0, 1, 0, 2, 2]]
	
	print(preprocess(input_matrix))