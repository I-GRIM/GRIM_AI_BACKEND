import argparse
import os
from hanspell import spell_checker


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text',  type=str,  default='./')
    args = parser.parse_args()
    
    output_path = "./"    
    # output_path = "/home/g0521sansan/korean_check/"
    os.remove(output_path+"spellList.txt")
    
    
    f = open(output_path+"spellList.txt","w")
    
    text = args.text
    text_part = text.split(" ")

    spelled_sent = spell_checker.check(text)
    checked_sent = spelled_sent.checked
    checked_sent_part = checked_sent.split(" ")

    parse_list = []

    for i,( o, c) in enumerate(zip(text_part, checked_sent_part)) :
        if(o!=c) :
            parse_list.append([i,o,c])
    for p in parse_list :
        f.write(str(p[0])+" "+p[1]+" "+p[2])
        f.write("\n")
        print(p)