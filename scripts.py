
def extract(infile, outfile, column, row, filling):
	with open(infile, 'r') as rd, open(outfile, 'w') as w:
		total = 0
		while total < row:
			line = rd.readline()
			if not line:
				break
			punc_idx = len(line)
			label = float(line[0:1])
			if label > 1:
				label = 1
			feature_line = line[2:punc_idx]
			words = feature_line.split(' ')
			if len(words) == 1:
				words = feature_line.split(',')
			cur_feature_list = line[0:1]
			if filling == True:
				matrix = column * [0.0000]
			else:
				matrix = column * [None]

			try:
				for word in words:
					if not word:
						continue
					tokens = word.split(':')
					if int(tokens[0]) < column:
						matrix[int(tokens[0])] = tokens[1]
					else:
						break

				for ix, va in enumerate(matrix):
					if va != None:
						cur_feature_list += ' ' + str(ix) + ':' + str(va)

				w.write(cur_feature_list+'\n')
			except:
				raise
				total -= 0
			
			total += 1
if __name__ == '__main__':
	extract('ele_test.txt', 'ele_mini_test.txt', 140, 10000000, True)
