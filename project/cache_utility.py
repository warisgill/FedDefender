import argparse
import os
from diskcache import Index


def printCacheFLConfigs(args):
    keys = []
    cache = Index(args.cache_dir)
    for k in cache.keys():
        if "-round:" not in k:
            d = cache[k]            
            if 'complete' in d.keys():
                keys.append(k)
    
    with open(args.out_txt, "w") as f:
        for k in keys:
            f.write(k)
            f.write("\n")

def printAllKeys(args):
    cache = Index(args.cache_dir)
    # with open(args.out_txt, "w") as f:
    for k in cache.keys():
        print(k)

def printKeysAndValues(args):
    cache = Index(args.cache_dir)
    for k in cache.keys():
        print(k, cache[k])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache utilty function")
    parser.add_argument("--cache_dir", type=str, default="", help="experiments cache directory", required=True)
    parser.add_argument("--out_txt", type=str, default="_exp_config_keys.txt", help="output text file")
    args =  parser.parse_args()  

    assert args.cache_dir.endswith("/"), "cache_dir should end with '/'" 
    assert os.path.exists(
        args.cache_dir), "cache dir does not exist"

    printAllKeys(args)
    printCacheFLConfigs(args)
    # printKeysAndValues(args)
    print("Done")