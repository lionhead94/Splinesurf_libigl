#! /usr/bin/env python3 -B

def clean():
    os.system('rm -rf bin && rm -rf build')
    
if __name__ == '__main__':
    clean()