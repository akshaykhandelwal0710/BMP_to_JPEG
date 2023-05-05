import numpy as np
import cv2
import re
from queue import PriorityQueue

samp = 2

cur = 0
def get_RGB(image_loc):
    global cur
    file = open(image_loc, 'rb')
    data = file.read()
    s = data.hex()
    
    l = re.findall('..?', s)
    cur = 0
    def get_next(cnt):
        global cur
        cur += cnt
        return l[cur - cnt: cur]
    def hex_to_int(lst, start, end):
        return int(''.join(lst[start:end][::-1]), 16)
    
    # Fetching File Header
    file_header = get_next(14)
    size = hex_to_int(file_header, 2, 6)
    data_offset = hex_to_int(file_header, 10, 14)
    
    # Fetching File Info Header
    info_header = get_next(40)
    width = hex_to_int(info_header, 4, 8)
    height = hex_to_int(info_header, 8, 12)
    bpp = hex_to_int(info_header, 14, 16)
    
    if (bpp != 24):
        print("File not in RGB format.")
        
    # Parsing the data
    cur = data_offset
    R = np.zeros((height, width), dtype = 'float32')
    G = np.zeros((height, width), dtype = 'float32')
    B = np.zeros((height, width), dtype = 'float32')
    for i in range(height-1, -1, -1):
        for j in range(width):
            b, g, r = get_next(1)[0], get_next(1)[0], get_next(1)[0]
            r, g, b = int(r, 16), int(g, 16), int(b, 16)
            R[i, j] = r
            G[i, j] = g
            B[i, j] = b
    return R, G, B

def reshape(R_old, G_old, B_old):
    s = R_old.shape
    new_shape = (((s[0] + 15) // 16) * 16, ((s[1] + 15) // 16) * 16)

    R = np.zeros(new_shape, dtype = 'float32')
    G = np.zeros(new_shape, dtype = 'float32')
    B = np.zeros(new_shape, dtype = 'float32')
    R[:s[0], :s[1]] = R_old
    G[:s[0], :s[1]] = G_old
    B[:s[0], :s[1]] = B_old
    return R, G, B

def RGB_to_YCbCr(R, G, B):
    global samp
    Y = 0.299*R + 0.587*G + 0.114*B
    cbb = 128 - 0.168736*R - 0.331264*G + 0.5*B
    crr = 128 + 0.5*R - 0.418688*G - 0.081312*B
    
    s = Y.shape
    s1 = (s[0] // samp, s[1] // samp)
    Cb = np.zeros(s1, dtype = float)
    Cr = np.zeros(s1, dtype = float)

    # average every samp x samp block for Cb and Cr
    for i in range(s[0]//samp):
        for j in range(s[1]//samp):
            x = i*samp
            y = j*samp
            avg = 0
            for dx in range(samp):
                for dy in range(samp):
                    avg += cbb[x + dx, y + dy]
            avg //= (samp * samp)
            Cb[i, j] = avg
            avg = 0
            for dx in range(samp):
                for dy in range(samp):
                    avg += crr[x + dx, y + dy]
            avg //= (samp * samp)
            Cr[i, j] = avg
            
    return Y, Cb, Cr

def perform_DCT(Y, Cb, Cr):
    global samp
    Y -= 128
    Cb -= 128
    Cr -= 128
    Y_blocks = []
    Cb_blocks = []
    Cr_blocks = []
    s1 = Cb.shape
    for i in range(0, s1[0], 8):
        if False:
            for j in range(s[1] - 8, -1, -8):
                grid = np.float32(Cb[i:i+8, j:j+8])
                Cb_blocks.append(cv2.dct(grid))
                grid = np.float32(Cr[i:i+8, j:j+8])
                Cr_blocks.append(cv2.dct(grid))
                grid = np.float32(Y[i*2:i*2+8, j*2:j*2+8])
                Y_blocks.append(cv2.dct(grid))
                grid = np.float32(Y[i*2:i*2+8, j*2+8:j*2+16])
                Y_blocks.append(cv2.dct(grid))
                grid = np.float32(Y[i*2+8:i*2+16, j*2:j*2+8])
                Y_blocks.append(cv2.dct(grid))
                grid = np.float32(Y[i*2+8:i*2+16, j*2+8:j*2+16])
                Y_blocks.append(cv2.dct(grid))
        else:
            for j in range(0, s1[1], 8):
                grid = np.float32(Cb[i:i+8, j:j+8])
                Cb_blocks.append(cv2.dct(grid))
                grid = np.float32(Cr[i:i+8, j:j+8])
                Cr_blocks.append(cv2.dct(grid))
                for dx in range(samp):
                    for dy in range(samp):
                        grid = np.float32(Y[i*samp+dx*8:i*samp+(dx+1)*8, j*samp+dy*8:j*samp+(dy+1)*8])
                        Y_blocks.append(cv2.dct(grid))
            
    return Y_blocks, Cb_blocks, Cr_blocks

def Quantize(data_blocks, quantization_table):
    new_blocks = []
    for block in data_blocks:
        new_blocks.append(np.around(block / quantization_table))
    return new_blocks

def spiral_traversal(block):
    traversal = [
        [1, 2, 6, 7, 15, 16, 28, 29],
        [3, 5, 8, 14, 17, 27, 30, 43],
        [4, 9, 13, 18, 26, 31, 42, 44],
        [10, 12, 19, 25, 32, 41, 45, 54],
        [11, 20, 24, 33, 40, 46, 53, 55],
        [21, 23, 34, 39, 47, 52, 56, 61],
        [22, 35, 38, 48, 51, 57, 60, 62],
        [36, 37, 49, 50, 58, 59, 63, 64]]

    arr = [0 for i in range(64)]
    for i in range(8):
        for j in range(8):
            arr[traversal[i][j]-1] = block[i, j]
    return arr

def get_cat(num): ##extracting position of msb to determine what length of bits it will need to be encoded
    num = int(abs(num))
    ans = 0
    pwr = 1
    while pwr <= num:
        pwr <<= 1
        ans += 1
    return ans

def to_two_bytes(num):
    bh = bin(num).replace("0b", "")
    bh = bh[::-1]
    while len(bh) < 16:
        bh += '0'
    bh = bh[::-1]

    result = [int(bh[:8], 2), int(bh[8:], 2)]
    return result

dc = 0
def Run_length_encoding(blocks):
    global dc

    dc = 0
    def encode(block):
        global dc
        temp = spiral_traversal(block)
        l = len(temp)

        encoded_block = []
        encoded_block.append((0, temp[0] - dc))
        dc = temp[0]
        c0 = 0
        cnz = 0
        for i in range(1, l):
            if temp[i]: cnz += 1
        for i in range(1, l):
            if cnz == 0:
                break
            if temp[i] or c0==15 :
                encoded_block.append((c0, temp[i]))
                c0 = 0
                if temp[i]:
                    cnz -= 1
            else:
                c0 += 1
        encoded_block.append((0, 0))

        encoded_block = np.array(encoded_block, dtype = int)
        return encoded_block
    
    encoded_blocks = []
    for block in blocks:
        encoded_blocks.append(encode(block))
    return encoded_blocks

def make_freq_table(encoded_blocks):
    dc_freq = np.zeros((1, 16), dtype=int)
    ac_freq = np.zeros((16, 16), dtype=int)
    for block in encoded_blocks:
        #block of length 64 usually (we take 8*8 grids), first element is dc coeff, the rest are ac coeffs
        #dc block[0] is encoded as (0, value)
        dc = block[0]
        dc_run_length, dc_val = dc #dc_run_length will be 0
        
        cat = get_cat(dc_val)
        dc_freq[dc_run_length, cat] += 1
        
        for i in range(1, len(block)):
            
            #ac block[i] is encoded as (run length of zeros before value max 15, value of ith non zero ac coefficient)
            ac_i = block[i]
            run_length, value = ac_i
            cat = get_cat(value) # find the category of value
            ac_freq[run_length, cat] += 1
            
    return dc_freq, ac_freq

depth = 0
def get_codebook(freq):
    global depth
    codebook = np.empty(freq.shape, dtype = object)
    
    def get_code_lengths(freq):

        def make_tree(freq):
            global code
            q = PriorityQueue()
            m = len(freq)
            n = len(freq[0])

            cnt = 0
            for i in range(m):
                for j in range(n):
                    if freq[i,j]:
                        q.put((freq[i,j], str((i,j))))
                        cnt += 1

            for i in range(cnt-1):
                a = q.get()
                b = q.get()
                q.put((a[0]+b[0], str("[" + a[1] + "," + b[1] + "]")))

            tree = eval(q.get()[1])
            return tree

        tree = make_tree(freq)
        code_lengths = np.zeros(freq.shape, dtype=int)

        depth = 0
        def get_depths(arr, code_lengths):
            global depth
            if type(arr[1]) == int and type(arr[0]) == int:
                code_lengths[arr[0], arr[1]] = min(depth, 16)
            else:
                depth += 1
                get_depths(arr[0], code_lengths)
                get_depths(arr[1], code_lengths)
                depth-=1

        get_depths(tree, code_lengths)
        return code_lengths
    
    code_lengths = get_code_lengths(freq)
    
    length_symbol_pairs = []
    
    m, n = freq.shape
    for i in range(m):
        for j in range(n):
            if code_lengths[i, j]:
                length_symbol_pairs.append((code_lengths[i, j], (i, j)))
    length_symbol_pairs.sort()
    
    def get_code(code, length):
        st = bin(code).replace("0b", "")[::-1]
        while len(st) < length:
            st += '0'
        return st[::-1]
    
    code = 0
    cur_len = 0
    last = length_symbol_pairs[-1]
    length_symbol_pairs[-1] = (last[0] + 1, last[1])
    for length, symbol in length_symbol_pairs:
        while cur_len < length:  #if there is no symbol with the current length
            cur_len += 1
            code *= 2
        codebook[symbol[0], symbol[1]] = get_code(code, length)
        code += 1
        
    return codebook, length_symbol_pairs

def get_huffman_codebooks(encoded_blocks):
    dc_huffman, ac_huffman = make_freq_table(encoded_blocks)
    dc_codebook, dc_symbol_pairs = get_codebook(dc_huffman)
    ac_codebook, ac_symbol_pairs = get_codebook(ac_huffman)
    
    return dc_codebook, dc_symbol_pairs, ac_codebook, ac_symbol_pairs

def parse_table(sorted_symbols, typ):
    table = [0xff, 0xc4]
    ln = 19
    symbols = []
    freqs = [0 for i in range(16)]
    
    for length, symbol in sorted_symbols:
        freqs[length-1] += 1
        symbols.append(symbol[0] * 16 + symbol[1])
        ln += 1
    
    table.extend(to_two_bytes(ln))
    table.append(typ)
    table.extend(freqs)
    table.extend(symbols)
    return table

def bit_rep(value):
    if value == 0:
        return ''
    else : 
        v = bin(abs(value)).replace("0b", '')
        if value>0 :
            return v
        else:
            return v.replace('0', 'a').replace('1','0').replace('a','1')

def encode_data(encoded_Y, encoded_Cb, encoded_Cr, codes_dc_Y, codes_dc_C, codes_ac_Y, codes_ac_C):
    global samp
    s = ''
    def encode_block(block, dc_codebook, ac_codebook):
        ans = ''
        dc_val = block[0][1]
        cat = get_cat(dc_val)
        ans+=(dc_codebook[0,cat]+bit_rep(dc_val))
        
        for i in range(1, len(block)):
            run, val = block[i]
            ans+=(ac_codebook[run, get_cat(val)]+bit_rep(val))
        return ans
    
    for i in range(len(encoded_Y)):
        s += encode_block(encoded_Y[i], codes_dc_Y, codes_ac_Y)
        if (i % (samp * samp) == samp * samp - 1):
            s += encode_block(encoded_Cb[(i) // (samp * samp)], codes_dc_C, codes_ac_C)
            s += encode_block(encoded_Cr[(i) // (samp * samp)], codes_dc_C, codes_ac_C)
        
    #pad with 0s
    a = len(s)%8
    s+='0'*a
    
    data = [int(s[i:i+8], 2) for i in range(0, len(s), 8)]
    fin_data = []
    for ele in data:
        fin_data.append(ele)
        if ele == 0xff:
            fin_data.append(0x00)
    return fin_data

def create_jpeg_data(shape, Y_q, C_q, sorted_symbols_dc_Y, sorted_symbols_ac_Y, sorted_symbols_dc_C, sorted_symbols_ac_C, encoded_Y, encoded_Cb, encoded_Cr):
    global samp
    jpeg_image = [0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10]

    header = [0x4a, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x01, 0x00, 0x48, 0x00, 0x48, 0x00, 0x00]
    jpeg_image.extend(header)

    luminance_quantisation_table = [0xff, 0xdb, 0x00, 0x43, 0x00]
    luminance_quantisation_table.extend([int(ele) for ele in spiral_traversal(Y_q)])
    jpeg_image.extend(luminance_quantisation_table)

    chrominance_quantisation_table = [0xff, 0xdb, 0x00, 0x43, 0x01]
    chrominance_quantisation_table.extend([int(ele) for ele in spiral_traversal(C_q)])
    jpeg_image.extend(chrominance_quantisation_table)

    start_of_frame = [0xff, 0xc0, 0x00, 0x11, 0x08]
    start_of_frame.extend(to_two_bytes(shape[0]))
    start_of_frame.extend(to_two_bytes(shape[1]))
    start_of_frame.extend([0x03, 0x01, 16 * samp + samp, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01])
    jpeg_image.extend(start_of_frame)
    
    jpeg_image.extend(parse_table(sorted_symbols_dc_Y, 0x00))
    jpeg_image.extend(parse_table(sorted_symbols_ac_Y, 0x10))
    jpeg_image.extend(parse_table(sorted_symbols_dc_C, 0x01))
    jpeg_image.extend(parse_table(sorted_symbols_ac_C, 0x11))
    
    start_of_scan = [0xff, 0xda, 0x00, 0x0c, 0x03, 0x01, 0x00, 0x02, 0x11, 0x03, 0x11, 0x00, 0x3f, 0x00]
    jpeg_image.extend(start_of_scan)
    
    jpeg_image.extend(encode_data(encoded_Y, encoded_Cb, encoded_Cr, codes_dc_Y, codes_dc_C, codes_ac_Y, codes_ac_C))

    end_of_image = [0xff, 0xd9]
    jpeg_image.extend(end_of_image)
    return jpeg_image

# Main
image = 'snail.bmp'

R_old, G_old, B_old = get_RGB(image)
old_shape = R_old.shape

# Reshaping to multiple of 16
R, G, B = reshape(R_old, G_old, B_old)
s = R.shape

# Conversion to Y Cb Cr colorspace
Y, Cb, Cr = RGB_to_YCbCr(R, G, B)

# Performing DCT and breaking each component into 8x8 blocks
Y_blocks, Cb_blocks, Cr_blocks = perform_DCT(Y, Cb, Cr)

# Quantisation Tables
Y_q = np.array([[4., 3, 4, 4, 4, 6, 11, 15], [3, 3, 3, 4, 5, 8, 14, 19], [3, 4, 4, 5, 8, 12, 16, 20], [4, 5, 6, 7, 12, 14, 18, 20], [6, 6, 9, 11, 14, 17, 21, 23], [9, 12, 12, 18, 23, 22, 25, 21], [11, 13, 15, 17, 21, 23, 25, 21], [13, 12, 12, 13, 16, 19, 21, 21]])
C_q = np.array([[4., 4, 6, 10, 21, 21, 21, 21], [4, 5, 6, 21, 21, 21, 21, 21], [6, 6, 12, 21, 21, 21, 21, 21], [10, 14, 21, 21, 21, 21, 21, 21], [21, 21, 21, 21, 21, 21, 21, 21], [21, 21, 21, 21, 21, 21, 21, 21], [21, 21, 21, 21, 21, 21, 21, 21], [21, 21, 21, 21, 21, 21, 21, 21]])

# Performing Quantization
Y_blocks = Quantize(Y_blocks, Y_q)
Cb_blocks = Quantize(Cb_blocks, C_q)
Cr_blocks = Quantize(Cr_blocks, C_q)

# Performing Run-length Encoding
encoded_Y = Run_length_encoding(Y_blocks)
encoded_Cb = Run_length_encoding(Cb_blocks)
encoded_Cr = Run_length_encoding(Cr_blocks)

# Generating Huffman encoding tables
encoded_C = list(encoded_Cb)
encoded_C.extend(list(encoded_Cr))
codes_dc_Y, sorted_symbols_dc_Y, codes_ac_Y, sorted_symbols_ac_Y = get_huffman_codebooks(encoded_Y)
codes_dc_C, sorted_symbols_dc_C, codes_ac_C, sorted_symbols_ac_C = get_huffman_codebooks(encoded_C)

# Creating jpeg byte data
jpeg_data = create_jpeg_data(old_shape, Y_q, C_q, sorted_symbols_dc_Y, sorted_symbols_ac_Y, sorted_symbols_dc_C, sorted_symbols_ac_C, encoded_Y, encoded_Cb, encoded_Cr)

# Storing the final JPEG Image
try:
    with open("my_image.jpeg", 'wb') as f:
        for byte in jpeg_data:
            f.write(byte.to_bytes(1, byteorder='big'))
except Exception as e:
    print(e)
    
# Displaying the jpeg image raw data