import os


class TxtFile:
    split = ' '
    note = '#'
    @staticmethod
    def read(path,ignore_first_line=False,split=None):
        with open(path,'r+',encoding='utf-8') as f:
            lines=f.readlines()
        lines=[ line.replace('\n','') for line in lines if not line.startswith(TxtFile.note)]
        if split is None:
            lines=[ line.split(TxtFile.split)  for line in lines]
        else:
            lines = [line.split(split) for line in lines]
        if ignore_first_line:
            return lines[1:]
        else:
            return lines


    @staticmethod
    def write(lines,path,split=None):
        assert type(lines)==list
        for i in range(len(lines)):
            if type(lines[i])==list:
                if split is None:
                    lines[i]=TxtFile.split.join([str(now) for now in lines[i]])
                else:
                    lines[i]=split.join([str(now) for now in lines[i]])
            else:lines[i]=str(lines[i])
        with open(path,"w+",encoding='utf-8') as f:
            f.write('\n'.join(lines))



class Dir:
    @staticmethod
    def mkdir(dir):
        if not os.path.exists(dir):
            os.mkdir(dir)
        return dir