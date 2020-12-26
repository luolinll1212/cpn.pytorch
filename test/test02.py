# -*- coding: utf-8 -*-

import hashlib

def stringtomd5(originstr):
    """将string转化为MD5"""
    signaturemd5 = hashlib.md5()
    signaturemd5.update(originstr)
    return signaturemd5.hexdigest()

if __name__ == '__main__':
    a = "a".encode("utf-8")
    out = stringtomd5(a)
    print(out)



