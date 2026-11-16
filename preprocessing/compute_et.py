import argparse
from pathlib import Path
import numpy as np
def qianreE(Rn, Ta, qa):

    lanmuda = 2.5008 * 1000000

    cp = 1005

    Rv = 461.5

    EF = lanmuda * qa / (lanmuda * qa + (cp * Rv * Ta**2)/lanmuda)



    E = EF * Rn

    return E

def main():
    p=argparse.ArgumentParser(description='Calculate SFE ET from temperature, specific humidity and net radiation arrays.')
    p.add_argument('--input',required=True,help='NPZ with Rn (W m-2), temp (K), shum (kg kg-1)')
    p.add_argument('--output',required=True);a=p.parse_args();z=np.load(a.input)
    et=qianreE(z['Rn'],z['temp'],z['shum'])*3600*24/(2.5008*1000000)
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True);np.save(target,et)
if __name__=='__main__':main()
