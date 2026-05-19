# -*- coding: utf-8 -*-
"""
Code released in conjunction with the 2026 ICDAR Competition in 
Text Recognition on Greek Squeezes (TROGS-26)

Call run_evaluations(pred_dir,gt_dir) to evaluate.
pred_dir should contain your method's transcription files
    Filenames should be {squeeze}_transcript.txt
    Contents are line by line character transcriptions
gt_dir should contain letter box files with ground truth

@author: Nicholas Howe  CC-BY 4.0
"""

import math
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from textdistance import levenshtein
import matplotlib.pyplot as plt

class Box:
    '''
    # simple class to keep track of coordinates for the four corners of a box
    # note corner order is CCW from NW corner, as stored in letter box file
    '''
    def __init__(self,xnw,ynw,xsw,ysw,xse,yse,xne,yne):
        self.xnw = xnw
        self.ynw = ynw
        self.xsw = xsw
        self.ysw = ysw
        self.xse = xse
        self.yse = yse
        self.xne = xne
        self.yne = yne
        
    def bd(self):
        return [self.xnw,self.ynw,self.xsw,self.ysw,self.xse,self.yse,self.xne,self.yne]

    def center(self):
        return((self.xnw+self.xne+self.xse+self.xsw)/4,(self.ynw+self.yne+self.yse+self.ysw)/4)
    
    def radius(self):
        return (math.sqrt((self.xnw-self.xse)*(self.xnw-self.xse)+(self.ynw-self.yse)*(self.ynw-self.yse))+math.sqrt((self.xne-self.xsw)*(self.xne-self.xsw)+(self.yne-self.ysw)*(self.yne-self.ysw)))/4
    
    def area(self):
        # uses top and left side, assuming orthogonal rectangle
        return math.sqrt(((self.xne-self.xnw)**2+(self.yne-self.ynw)**2)*((self.xsw-self.xnw)**2+(self.ysw-self.ynw)**2))

    def angle(self):
        return math.atan2(self.yne-self.ynw,self.xne-self.xnw)

    def dist(self,box):
        c = self.center()
        c2 = box.center()
        return math.sqrt((c[0]-c2[0])*(c[0]-c2[0])+(c[1]-c2[1])*(c[1]-c2[1]))

    def plot(self,clr='r-'):
        plt.plot([self.xnw,self.xne,self.xse,self.xsw,self.xnw],[self.ynw,self.yne,self.yse,self.ysw,self.ynw],clr)
        
      
def convert(stringin):
    """
    Convert Greek characters to letter proxies.

    Parameters
    ----------
    stringin : String
        Character string possibly containing Greek.

    Returns
    -------
    txt : String
        Converted string with Greek characters replaced by Latin proxy.

    """
    eqv = {
        913:"A",  # ALPHA
        7944:"A", # ALPHA + AIGU
        914:"B",  # BETA
        915:"G",  # GAMMA
        916:"D",  # DELTA
        917:"E",  # EPSILON
        918:"Z",  # ZETA
        919:"H",  # ETA
        920:"Q",  # THETA
        921:"I",  # IOTA
        922:"K",  # KAPPA
        923:"L",  # LAMDA
        924:"M",  # MU
        925:"N",  # NU
        926:"X",  # XI
        927:"O",  # OMICRON
        928:"P",  # PI
        929:"R",  # PI
        931:"S",  # SIGMA
        932:"T",  # TAU
        933:"U",  # UPSILON
        934:"F",  # PHI
        935:"Y",  # PSI
        936:"C",  # CHI
        937:"W",  # OMEGA
        945:"a",  # ALPHA
        8118:"a",  # ALPHA + TILDE
        8119:"a",  # ALPHA + TILDE + CEDILLE
        7942:"a",  # ALPHA + TILDE + TICK
        7943:"a",  # ALPHA + TILDE + BACKTICK
        8070:"a",  # ALPHA + TILDE + TICK + CEDILLE
        8071:"a",  # ALPHA + TILDE + BACKTICK + CEDILLE
        8048:"a",  # ALPHA + GRAVE
        8049:"a",  # ALPHA + AIGU
        7936:"a",  # ALPHA + TICK
        7937:"a",  # ALPHA +BACKTICK
        7940:"a",  # ALPHA + TICK + AIGU
        7941:"a",  # ALPHA + BACKTICK + AIGU
        7938:"a",  # ALPHA + TICK + GRAVE
        7939:"a",  # ALPHA + BACKTICK + GRAVE
        946:"b",  # BETA
        947:"g",  # GAMMA
        948:"d",  # DELTA
        949:"e",  # EPSILON
        7952:"e",  # EPSILON + TICK
        7953:"e",  # EPSILON + BACKTICK
        8050:"e",  # EPSILON + GRAVE
        8051:"e",  # EPSILON + AIGU
        7956:"e",  # EPSILON + TICK + AIGU
        7957:"e",  # EPSILON + BACKTICK + AIGU
        7954:"e",  # EPSILON + TICK + GRAVE
        7955:"e",  # EPSILON + BACKTICK + GRAVE
        950:"z",  # ZETA
        951:"h",  # ETA
        7974:"h",  # ETA + TILDE + TICK + CEDILLE
        7975:"h",  # ETA + TILDE + BACKTICK + CEDILLE
        8086:"h",  # ETA + TILDE + TICK
        8087:"h",  # ETA + TILDE + BACKTICK
        8134:"h",  # ETA + TILDE
        8135:"h",  # ETA + TILDE + CEDILLE
        7970:"h",  # ETA + TICK + GRAVE
        7971:"h",  # ETA + BACKTICK + GRAVE
        8016:"h",  # ETA + TICK?
        8020:"h",  # ETA + TICK + AIGU
        7968:"h",  # ETA + TICK
        7969:"h",  # ETA + BACKTICK
        7972:"h",  # ETA + TICK + AIGU
        7973:"h",  # ETA + BACKTICK + AIGU
        8052:"h",  # ETA + GRAVE
        8053:"h",  # ETA + AIGU ?
        952:"q",  # THETA
        953:"i",  # IOTA
        7990:"i",  # IOTA + TILDE + TICK
        7991:"i",  # IOTA + TILDE + BACKTICK
        8150:"i",  # IOTA + TILDE
        8054:"i",  # IOTA + GRAVE
        8055:"i",  # IOTA + AIGU
        7984:"i",  # IOTA + TICK
        7985:"i",  # IOTA + BACKTICK
        7988:"i",  # IOTA + TICK + AIGU
        7989:"i",  # IOTA + BACKTICK + AIGU
        7986:"i",  # IOTA + TICK + GRAVE
        7987:"i",  # IOTA + BACKTICK + GRAVE
        954:"k",  # KAPPA
        955:"l",  # LAMDA
        956:"m",  # MU
        957:"n",  # NU
        834:"n",  # NU + TILDE
        958:"x",  # XI
        959:"o",  # OMICRON
        8056:"o",  # OMICRON + GRAVE
        8057:"o",  # OMICRON + AIGU
        8000:"o",  # OMICRON + TICK
        8001:"o",  # OMICRON + BACKTICK
        8004:"o",  # OMICRON + TICK + AIGU
        8005:"o",  # OMICRON + BACKTICK + AIGU
        8002:"o",  # OMICRON + TICK + GRAVE
        8003:"o",  # OMICRON + BACKTICK + GRAVE
        960:"p",  # PI
        961:"r",  # RHO
        962:"v",  # SIGMA
        963:"s",  # SIGMA
        964:"t",  # TAU
        965:"u",  # UPSILON
        8022:"u",  # UPSILON + TILDE + TICK
        8023:"u",  # UPSILON + TILDE + BACKTICK
        8166:"u",  # UPSILON + TILDE
        8016:"u",  # UPSILON + TICK
        8017:"u",  # UPSILON + BACKTICK
        8020:"u",  # UPSILON + TICK + AIGU
        8021:"u",  # UPSILON + BACKTICK + AIGU
        8018:"u",  # UPSILON + TICK + GRAVE
        8019:"u",  # UPSILON + BACKTICK + GRAVE
        8058:"u",  # UPSILON + GRAVE
        8059:"u",  # UPSILON + AIGU ?
        966:"f",  # PHI
        967:"y",  # PSI
        968:"c",  # CHI
        969:"w",  # OMEGA
        8102:"w",  # OMEGA + TILDE + TICK + CEDILLE
        8103:"w",  # OMEGA + TILDE + BACKTICK + CEDILLE
        8038:"w",  # OMEGA + TILDE + TICK
        8039:"w",  # OMEGA + TILDE + BACKTICK
        8182:"w",  # OMEGA + TILDE
        8183:"w",  # OMEGA + TILDE + CEDILLE
        8060:"w",  # OMEGA + GRAVE
        8061:"w",  # OMEGA + AIGU
        8032:"w",  # OMEGA + TICK
        8033:"w",  # OMEGA + BACKTICK
        8036:"w",  # OMEGA + TICK + AIGU
        8037:"w",  # OMEGA + BACKTICK + AIGU
        8034:"w",  # OMEGA + TICK + GRAVE
        8035:"w",  # OMEGA + BACKTICK + GRAVE
    
        8217:"'",  # TICK???
        8025:"`",  # BACKTICK???
        803:"_",  # CEDILLE???
        8195:" ",  # WEIRD SPACE???
        903:".",  # WEIRD DOT???
        8228:".",  # WEIRD DOT???
        8311:"7",  # SUPERSCRIPT 7
    
        # Formatting characters
        10:"\n",  # CARRIAGE RETURN
        3:" ",  # SPACE
        32:" "  # SPACE
    }
    for i in range(0, 254):
        eqv[i] = chr(i)
            
    txt = []
    for char in stringin:
        if ord(char) in eqv:
            txt.append(eqv[ord(char)])
        else:
            txt.append(char)               
    txt = "".join(txt)
    return txt.upper()


def readBoxFile(fname):
    """
    Parameters
    ----------
    fname : Name of the file to read

    Returns
    -------
    angle : Primary angle of the text
    boxes : List of boxes
    transcript : Box annotations
    lines : Information about how boxes are arranged in lines
    """
    angle = None
    boxes = []
    transcript = []
    lines = []
    try:
        with open(fname, "r") as f:
            phase = 0
            count = 0
            for line in f:
                count = count+1
                line2 = line.replace('*','').replace(' ','')
                if line2=='' or line2=='\n':
                    continue  # skip empty line
                elif line[0]=='#':
                    phase = phase+1
                elif phase==0:  # File should begin with a comment
                    #if line[0]=='# Baseline angle\n':
                    print("Header problem in file {}.".format(fname))
                elif phase==1:
                    #if line=='# Boxes: x1 y1 x2 y2 x3 y3 x4 y4\n':
                    if angle==None:
                        angle = float(line)
                    else:
                        print("Multiple angle data found in {}.".format(fname))
                elif phase==2:
                    #if line=='# Transcript\n':
                    n = line.split()
                    if len(n)==8:
                        box = Box(*[float(c) for c in n])
                        if box.area() < 9:
                            print("Skipping box with very small area (line {}).".format(count))
                        else:
                            boxes.append(box)
                    else:
                        print('Incorrect parameter count for box line ({}) in {}.'.format(len(n),fname))   
                elif phase==3:
                    transcript.append(line.replace('*','').replace('\n','').replace('\r','').replace(' ',''))
                elif phase==4:
                    n = line.split()
                    if len(n)==4:
                        lines.append([float(c) for c in n])
                        if np.any(np.isnan(lines[-1])):
                            print('Bad values found in line specification (file line {}) -- skipping.'.format(count))
                            lines.pop()
                        elif lines[-1][0]==lines[-1][2] and lines[-1][1]==lines[-1][3]:
                            print('Zero-length line found in line specification (file line {}) -- skipping.'.format(count))
                            lines.pop()
                    else:
                        print('Incorrect parameter count for line specifiers ({}) in {}.'.format(len(n),fname))   
            f.close()
    except FileNotFoundError:
        print(f"File '{fname}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    return angle,boxes,transcript,lines


def dist2seg(x,y,seg):
    '''
    # compute the distance from a point to the nearest point on a line segment
    '''
    x0 = seg[0]
    y0 = seg[1]
    x1 = seg[2]
    y1 = seg[3]
    rx = x-x0
    ry = y-y0
    rx1 = x1-x0
    ry1 = y1-y0
    seglen = math.sqrt(rx1**2+ry1**2)
    parcomp = (rx*rx1+ry*ry1)/seglen
    if parcomp<0:
        # closest point is p0 endpoint
        d = math.sqrt(rx**2+ry**2)
    elif parcomp > seglen:
        # closest point is p1 endpoint
        d = math.sqrt((x-x1)**2+(y-y1)**2)
    else:
        # find closest point p along segment
        t = parcomp/seglen
        px = x0+t*rx1
        py = y0+t*ry1
        d = math.sqrt((x-px)**2+(y-py)**2)
    return d


def getRowBoxes(boxes,gangle=None,lines=[]):
    """
    Orders a set of boxes into rows.
    Designed to work with data from a letter box file.
    """
    if len(lines)==0:
        # infer lines by spatial arrangement
        nboxes = len(boxes)
        bclust = np.full((nboxes),-1)
        bseq = np.zeros((nboxes))
        lnbr = np.full((nboxes),-1)
        rnbr = np.full((nboxes),-1)
        
        # speed up inner loop by sorting x coordinates
        xcent = [b.center()[0] for b in boxes]
        xsort = np.sort(xcent)
        xord = np.argsort(xcent)
        xlut = np.argsort(xord)
    
        # find left and right nearest neighbors for each box
        for i in range(nboxes):
            ci = boxes[i].center()
            if (gangle==None):
                # no global angle so compute a local one from the top line of this box
                angle = boxes[i].angle()
            else:
                angle = gangle
            ax = math.cos(angle)
            ay = math.sin(angle)
            rd = np.inf  # current distance to closest right leighbor
            ld = np.inf  # current distance to closest left neighbor
            # we are going to start at the x value of the current box, and check nearby
            # values of x both higher and lower
            klo = xlut[i]
            khi = klo
            assert(ci[0] == xsort[klo])
            while min(ci[0]-xsort[klo],xsort[khi]-ci[0]) < max(rd,ld) and (klo>0 or khi < nboxes-1):
                if (ci[0]-xsort[klo] < xsort[khi]-ci[0] or khi >= nboxes-1) and klo > 0:
                    # reduce klo
                    klo = klo-1
                    j = xord[klo]  # next closest x value below
                elif khi < nboxes-1:
                    # increase khi
                    khi = khi+1
                    j = xord[khi]  # next closest x value above
                else:
                    print('Should not reach this!')
                # check whether box j is closer to i on either the left or the right
                d = boxes[i].dist(boxes[j])
                cj = boxes[j].center()
                rj = boxes[j].radius();
                rjsq = rj*rj
                dcx = cj[0]-ci[0]
                dcy = cj[1]-ci[1]
                dotprod = dcx*ax+dcy*ay
                isRight = dotprod > 0
                if (i!=j) and (d<ld) and not isRight:
                    # found a closer left neighbor, so check position and update if near line
                    c = (ci[0]-d*ax, ci[1]-d*ay)
                    if (cj[0]-c[0])**2+(cj[1]-c[1])**2 < rjsq:
                        ld = d
                        lnbr[i] = j
                if (i!=j) and (d<rd) and isRight:
                    # found a closer right neighbor, so check position and update if near line
                    c = (ci[0]+d*ax, ci[1]+d*ay)
                    if (cj[0]-c[0])**2+(cj[1]-c[1])**2 < rjsq:
                        rd = d
                        rnbr[i] = j
                #print(i,j)
        
        # sanity/correctness check for neighbor detection -- hopefully should never report a mismatch.
        for i in range(len(rnbr)):
            if rnbr[i]>=0 and lnbr[rnbr[i]]!=i:
                if lnbr[rnbr[i]] >= 0 and rnbr[lnbr[rnbr[i]]] == rnbr[i]:
                    # pointing to a neighbor that is mutually satisfied; this link is probably wrong
                    rnbr[i] = -2
            if lnbr[i]>=0 and rnbr[lnbr[i]]!=i:
                if rnbr[lnbr[i]] >= 0 and lnbr[rnbr[lnbr[i]]] == lnbr[i]:
                    # pointing to a neighbor that is mutually satisfied; this link is probably wrong
                    lnbr[i] = -2
        for i in range(len(rnbr)):
            if rnbr[i]>=0 and lnbr[rnbr[i]]!=i:
                if lnbr[rnbr[i]]<0:
                    # link up unassigned box
                    lnbr[rnbr[i]] = i
                else:
                    print('Left of right mismatch at {}: {} to {}.'.format(i,rnbr[i],lnbr[rnbr[i]]))
                    #lnbr[rnbr[i]] = -3
                    #rnbr[i]=-3
            if lnbr[i]>=0 and rnbr[lnbr[i]]!=i:
                if rnbr[lnbr[i]]<0:
                    # link up unassigned box
                    rnbr[lnbr[i]] = i
                else:
                    print('Right of left mismatch at {}: {} to {}.'.format(i,lnbr[i],rnbr[lnbr[i]]))
                    #rnbr[lnbr[i]] = -3
                    #lnbr[i] = -3
    
        # cluster boxes by row: combine nearest neighbors one by one
        clust = 0
        cid = []
        cseq = []
        for i in range(nboxes):
            if (gangle==None):
                angle = boxes[i].angle()
            else:
                angle = gangle
            if (bclust[i] == -1):
                # we have found a box belonging to a new cluster, so assemble all of its neighbors and give them the same cluster label
                c = [i]
                bclust[i] = clust
                bseq[i] = boxes[i].center()[0]*math.cos(angle)+boxes[i].center()[1]*math.sin(angle)
                j = lnbr[i]
                while (j>=0) and not j in c:
                    bclust[j] = clust
                    c.append(j)
                    bseq[j] = boxes[j].center()[0]*math.cos(angle)+boxes[j].center()[1]*math.sin(angle)
                    j = lnbr[j]
                j = rnbr[i]
                while (j>=0) and not j in c:
                    bclust[j] = clust
                    c.append(j)
                    bseq[j] = boxes[j].center()[0]*math.cos(angle)+boxes[j].center()[1]*math.sin(angle)
                    j = rnbr[j]
                cseq.append(-boxes[i].center()[0]*math.sin(angle)+boxes[i].center()[1]*math.cos(angle))
                cid.append(c)
                clust = clust+1
            # the new cluster is now labeled
            #print(clust,c,cseq,bseq)
        
        rowlist = []
        # at the end of this loop, all the boxes are labeled with a unique cluster id.
        # they may not be sorted left to right or top to bottom yet
        # cseq holds a value that increases from top to bottom, 
        # and bseq holds a value that increases from left to right within a row
        # so the boxes should be reordered and assembled according to these numbers.
        # use ord2lut to convert the numeric values to a list of indices that are in the correct order
        for c in cid:
            # c is a list of the box ids belonging to this cluster
            rowlist.append([boxes[c[i]] for i in np.argsort(bseq[c])])
        return [rowlist[i] for i in np.argsort(cseq)], [cid[i] for i in np.argsort(cseq)]  # reorder rows based on cseq value
    else:
        # attribute boxes to lines based on supplied line information
        # first, identify any polylines:
        label = np.zeros(len(lines),dtype=np.int32)
        for i in range(1,len(lines)):
            if lines[i][0]==lines[i-1][2] and lines[i][1]==lines[i-1][3]:
                # we have a continuation
                label[i] = label[i-1]
            else:
                # new polyline
                label[i] = label[i-1]+1

        # next, find min distance from each box to a line segment
        blabel = np.zeros(len(boxes))
        for i in range(len(boxes)):
            d = dist2seg(*boxes[i].center(),lines[0])
            blabel[i] = 0
            for j in range(1,len(lines)):
                jd = dist2seg(*boxes[i].center(),lines[j])
                if jd<d:
                    d = jd
                    blabel[i] = label[j]
                    
        # now arrange the boxes within the rows
        rowlist = []
        idlist = []
        for i in range(label[-1]+1):
            j = 0
            while j < len(boxes) and blabel[j]!=i:  # keep going until we find the first box with label i
                j = j+1
            if j<len(boxes):  # just make sure we found at least one box assigned to this label
                row = [boxes[j]]
                rowid = [j]
                rank = [0]
                for k in range(j+1,len(boxes)):
                    if blabel[k]==i:
                        row.append(boxes[k])
                        rowid.append(k)
                        rank.append((boxes[k].xnw-boxes[j].xnw)*(boxes[j].xne-boxes[j].xnw)+(boxes[k].ynw-boxes[j].ynw)*(boxes[j].yne-boxes[j].ynw))
                # now we have all the boxes in this row, and rank holds a value that will sort them
                rowlist.append([row[i] for i in np.argsort(rank)])
                idlist.append([rowid[i] for i in np.argsort(rank)])
        return rowlist, idlist
    
    
def orderBoxes(boxes,gangle=None,lines=[]):
    rowlist, idlist = getRowBoxes(boxes,gangle=gangle,lines=lines)
    return [b for row in rowlist for b in row]


def getRowTranscript(fname):
      """
      Reads a file of annotated letter boxes and arranges them into rows 
      ordered from left to right.
     
      For contest grading purposes, we want to use simple left-to-right,
      top down reading order, explicitly ignoring multicolumn layouts.
      
      To make sure the transcript matches, we need to first generate the 
      multicolumn result and then reorder into full-width lines.
      """   
      gangle, boxes, transcript, lines = readBoxFile(fname)
      boxes2 = orderBoxes(boxes,gangle,lines)  # reordered to transcript order
      rowlist, idlist = getRowBoxes(boxes2,gangle=gangle)  # ignore multicolumn
      lintran = ''.join(transcript)
      return ''.join([''.join([lintran[i] for i in rowid])+'\n' for rowid in idlist])
      

def evaluate(prediction,gt):
    """
    Evaluates the given prediction against the 
    corresponding ground truth file.
    Includes line breaks as characters.

    Parameters
    ----------
    prediction : String
        Text transcription to be graded
    gt : String
        Accepted ground truth transcript

    Returns
    -------
    CER, n_error, n_char

    """
    n_error = levenshtein(convert(prediction).upper().replace(' ','').replace('\t',''),gt)
    n_char = len(gt)
    cer = n_error/n_char
    return cer, n_error, n_char


def run_evaluations(pred_dir,gt_dir):
    """
    Runs the evaluation for all files in the ground truth directory.
    For a given ground truth file (named SQUEEZE_letters.txt), 
    it looks for a prediction file named SQUEEZE_pred.txt
    containing the transcribed letters.

    Parameters
    ----------
    pred_dir : String, optional
        Directory containing prediction files. The default is '.'.
    gt_dir : TYPE, optional
        Directory containing ground truth files. The default is '.'.

    Returns
    -------
    Mean character error rate.

    """
    gt_files = [f for f in listdir(gt_dir) if isfile(join(gt_dir, f)) and f[-29:]=='_Rotation1_300dpi_letters.txt']  # transcript is same for both rotations
    cer = []
    n_error = []
    n_char = []
    for gtf in gt_files:
        predf = os.path.join(pred_dir,gtf[:-29]+'_transcript.txt')
        try:
            with open(predf, "r", encoding='utf8') as f:
                pred = "".join([line for line in f])
        except FileNotFoundError:
            print(f"Unable to find prediction file '{predf}'.")
            pred = ""
        except Exception as e:
            print(f"An error occurred: {str(e)} while reading '{predf}'.")
            pred = ""
        pred = convert(pred)
        gt = getRowTranscript(os.path.join(gt_dir,gtf))
        sq_cer, sq_n_error, sq_n_char = evaluate(pred,gt)
        cer.append(sq_cer)
        n_error.append(sq_n_error)
        n_char.append(sq_n_char)
    return sum(n_error)/sum(n_char)


if __name__ == "__main__":
    #cer = run_evaluations(pred_dir='.',gt_dir='.')
    cer = run_evaluations(pred_dir='pred',gt_dir='gt')
    print(f'Mean character error rate is {cer}.')
