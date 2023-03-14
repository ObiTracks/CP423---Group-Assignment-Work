import argparse

def gamma_encode(n):
    if n == 0:
        return "0"
    b, pre = bin(n)[2:], "0" * (len(bin(n)) - 2)
    return pre + b

def delta_encode(n):
    if n == 0:
        return "0"
    b = bin(n)[2:]
    pre,suf = gamma_encode(len(b)), b[1:]
    return pre + suf

def gamma_decode(c):
    p = c.find("1")
    if p == -1:
        return None
    n = int("1" + c[p:], 2)
    return n

def delta_decode(c):
    p = c.find("1")
    if p == -1:
        return None
    pre = c[:p+1]
    l = gamma_decode(pre)
    suf = c[p+1:]
    if l is None or len(suf) < l:
        return None
    s = "1" + suf[:l-1]
    n = int(s, 2)
    return n

def encode(numbers, alg):
    encode_function = gamma_encode if alg == "gamma" else delta_encode
    codes = [encode_function(n) or "ERROR" for n in numbers]
    print(codes)

def decode(codes, alg):
    decode_function = gamma_decode if alg == "gamma" else delta_decode
    numbers = [decode_function(c) or "ERROR" for c in codes]
    print(numbers)

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--alg", required=True)
    mutually_exclusive_group = argument_parser.add_mutually_exclusive_group(required=True)
    mutually_exclusive_group.add_argument("--encode", nargs="+", type=int)
    mutually_exclusive_group.add_argument("--decode", nargs="+")
    parsed_arguments = argument_parser.parse_args()

    if parsed_arguments.encode is not None:
        encode(parsed_arguments.encode, parsed_arguments.alg)
    else:
        decode(parsed_arguments.decode, parsed_arguments.alg)
