import argparse
from pathlib import Path
import numpy as np
def shum_seitch_VPD(temp, shum, pres):

    VPD = 6.11*np.exp((17.27*(temp-273.15))/((temp-273.15)+237.3))-(shum*pres/100)/(0.622 + 0.378 * shum)      

    return VPD

def main():
    p=argparse.ArgumentParser(description='Calculate VPD from temperature, specific humidity and pressure arrays.')
    p.add_argument('--input',required=True,help='NPZ with temp (K), shum (kg kg-1), pres (Pa)')
    p.add_argument('--output',required=True);a=p.parse_args();z=np.load(a.input)
    vpd=shum_seitch_VPD(z['temp'],z['shum'],z['pres']);vpd[vpd>1000]=-9999
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True);np.save(target,vpd)
if __name__=='__main__':main()
