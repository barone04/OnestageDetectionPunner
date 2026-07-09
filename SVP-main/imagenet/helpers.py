import re


def get_cpr(compress_rate):
    cprate_str = compress_rate
    cprate_str_list = cprate_str.split("+")
    pat_cprate = re.compile(r"\d+\.\d*")
    pat_num = re.compile(r"\*\d+")
    cprate = []
    for x in cprate_str_list:
        num = 1
        find_num = re.findall(pat_num, x)
        if find_num:
            assert len(find_num) == 1
            num = int(find_num[0].replace("*", ""))
        find_cprate = re.findall(pat_cprate, x)
        assert len(find_cprate) == 1
        cprate += [float(find_cprate[0])] * num

    return cprate
