"""
MIT No Attribution

Copyright (c) 2025 Maura Putzer

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Filename: make_xml.py
Author: Maura Putzer
Last Updated: 2025-03-27
Description: This file converts .txt files with Greek inscription annotation information into .xml format for document analysis. 
"""

import os
import textwrap
import xml.etree.cElementTree as ET
from datetime import datetime, date
from PIL import Image
import argparse
import numpy as np
from scipy.spatial import ConvexHull
from collections import defaultdict

'''
Usage instructions for make_xml.py calling in command line
'''
def calling_instructions():
    print("--------------------")
    print("For detailed instructions on usage, command-line arguments, and errors: python3 make_xml.py -h/--help")
    print("CALLING IN THE COMMAND LINE:")
    print("\nPOSITIONAL ARGUMENTS:")
    print("To create one XML file from a directory: python3 make_xml.py FILENAME.txt *LETTER_BOXES_DIRECTORY *PNG_IMAGES_DIRECTORY *XML_STORAGE_DIRECTORY, *IS_GREEK")
    print("To create all XML files from a directory: python3 make_xml.py *LETTER_BOXES_DIRECTORY *PNG_IMAGES_DIRECTORY *XML_STORAGE_DIRECTORY, *IS_GREEK")
    print("\nKEYWORD ARGUMENTS (arguments can be in any order):")
    print("To create one XML file from a directory: python3 make_xml.py -s FILENAME.txt -o *LETTER_BOXES_DIRECTORY -im *PNG_IMAGES_DIRECTORY -xml *XML_STORAGE_DIRECTORY -greek *IS_GREEK")
    print("To create all XML files from a directory: python3 make_xml.py -o *LETTER_BOXES_DIRECTORY -im *PNG_IMAGES_DIRECTORY -xml *XML_STORAGE_DIRECTORY -greek *IS_GREEK")
    print("\nBLEND KEYWORD/POSITIONAL ARGUMENTS:")
    print("You can also make calls with positional arguments followed by keyword arguments (ex: 'python3 make_xml.py FILENAME.txt -im *PNG_IMAGES_DIRECTORY' or 'python3 make_xml.py FILENAME.txt *LETTER_BOXES_DIRECTORY -xml *XML_STORAGE_DIRECTORY -im *PNG_IMAGES_DIRECTORY')")
    print("In these cases, the arguments without keywords must be in the order they appear (ex: FILENAME.TXT before LETTER_BOXES_DIRECTORY) and must appear BEFORE the keyword arguments")
    print("\n*Optional: defaults to current working directory (or False for IS_GREEK) if no argument provided.")
    print("--------------------")

'''
Greek dictionary
Converting to Greek chars is a problem because one letter can be many greek letters...
'''
eqv = {
        'A': 'α',

        'B': 'β',

        'C': 'χ', 
        
        'D': 'δ',

        'E': 'ε',

        'F': 'φ',

        'G': 'γ',  

        'H': 'η', 
        
        'I': 'ι', 

        'K': 'κ', 

        'L': 'λ', 

        'M': 'μ', 

        'N': 'ν', 

        'O': 'ο', 

        'P': 'π', 

        'Q': 'θ', 

        'R': 'ρ',  

        'S': 'σ',

        'T': 'τ', 

        'U': 'ψ', 

        'W': 'ω', 

        'X': 'ξ', 

        'Y': 'υ', 

        'Z': 'ζ', 
}

'''
Reads a specific line number in a given file
Args:
- filename (string)
- line (int): line number
Returns:
- line_data (string): the line of text
'''
def read_line(filename, line, do_strip):
    file = open(filename)
    content = file.readlines()
    line_data = content[line]
    #strips new line created
    if do_strip == True:
        line_data = line_data.rstrip("\n")
    return line_data

'''
Determines when a word appears in a file and returns the line number
Args:
- filename (file)
- word (string): word being searched for
Returns:
- line number (int): line number with the given word in the file
'''
def word_appears(filename, target_word):
    file = open(filename)
    line_number = 0
    for line in file:
        line_number += 1
        if target_word in line:
            return line_number

'''
Replaces certain occurrences of a a character
Args:
- sub (string): The string that must be replaced
- repl (string): The string that replaces sub
- txt (string): The string being modified
- nth (char): The character number being changed
Returns:
- new string with sub replaced
'''
#https://stackoverflow.com/questions/35091557/replace-nth-occurrence-of-substring-in-string
def replace_nth(sub,repl,txt,nth):
    arr=txt.split(sub)
    part1=sub.join(arr[:nth])
    part2=sub.join(arr[nth:])
    return part1+repl+part2

'''
Creates a list of the letters in the transcript
Args:
- ref_txt_file (file): the reference text file
- line_transcript (int): int for where the transcript line is
Returns:
- transcript_list (list): list of letters in transcript
'''
def create_transcript_list(ref_txt_file, line_transcript, is_greek):
    #looks at all the letters in transcript of old file and adds them to a list
    with open(ref_txt_file, 'r') as transcribed_file:
        transcript_list = []
        #increments lines so it is clear how many have been checked
        line_counter = line_transcript
        #Determines the last line with characters in the file
        max_line_num = len([l for l in transcribed_file.readlines() if l.strip(' \n') != ''])
        #while there are still lines in the transcript to check
        while (line_counter < max_line_num):
            #Reads transcript line
            transcript_data = read_line(ref_txt_file, line_counter, False)
            #Determines each character in the line and adds to the transcript list
            for i, letter in enumerate(transcript_data):
                char = transcript_data[i] 
                if is_greek and char in eqv:
                    transcript_list.append(eqv[char])
                elif (char != "*" and char != " "):
                    #Newline character not recognized by all os, so converted to recognizable term
                    if (char == "\n"):
                        char = "$"
                    #the first character in transcript list is not a new line
                    transcript_list.append(char)
            line_counter += 1
    print(transcript_list)
    return transcript_list

'''
Creates a dictionary with text regions, lines for those text regions, and bbox for entire text region
Args:
- line_transcript (int): int for where the transcript line is
- transcript_list (list): all glyphs (including new lines) in file
- ref_txt_file (file): the reference text file
- baseline_angle_data (string): baseline angle for the file
Returns:
- region_dict (dict): Dictionary with text regions, lines for those text regions, and bbox for entire text region
'''
def find_regions(line_transcript, transcript_list, ref_txt_file, baseline_angle_data):
    #Determines if lines exist after the marker for columns
    lines_after = False
    found = False
    for line in ref_txt_file:
        if line.strip() == "# Lines: x1 y1 x2 y2 x3 y3 x4 y4":
            found = True
            break
    if found:
        next_line = ref_txt_file.readline()
        if next_line:
            if next_line.strip():
                #Characters exist after the target line
                lines_after = True 
            else:
                #Only whitespace after target line
                lines_after = False 
        else:
            #No more lines after target line
            lines_after = False 

    region_dict = defaultdict(lambda: {'lines': [], 'coords': []})
    line_list = []
    coord_list = np.array([[]])
    #rotated_bboxes_line_dict where line bounding box for region ordering has been rotated
    bboxes_line_dict = bbox_for_group(line_transcript, transcript_list, "TextLineNStr", ref_txt_file, baseline_angle_data)
    baseline_angle_data = np.radians(float(baseline_angle_data))
    if lines_after == True:
        y1 = float('-inf')
        region_count = 1
        for line in bboxes_line_dict:
            #If value contained at that line number
            # print(line, "", bboxes_line_dict[line])
            if (len(bboxes_line_dict[line]) != 1):
                #If previous value greater than the next line value
                if np.max(bboxes_line_dict[line][:, 1]) < y1:
                    #Find bounding box that encompasses all the lines within a region
                    #Unrotate coord_list
                    rotation_matrix = np.array([[np.cos(baseline_angle_data), np.sin(baseline_angle_data)],[-np.sin(baseline_angle_data),np.cos(baseline_angle_data)]])
                    coord_list = np.matmul(coord_list, rotation_matrix)
                    hull = ConvexHull(coord_list)
                    vertices = hull.vertices
                    coords_region = coord_list[vertices]  
                    coords_region = np.round(coords_region).astype(int)
                    coords_region = ' '.join(map(str, coords_region.tolist()))
                    coords_region=coords_region.replace(", ", ",")
                    coords_region=coords_region.replace("[","")
                    coords_region=coords_region.replace("]","") 
                    region_dict[region_count]["coords"]  = coords_region
                    #Empty line_list and coord list for next region
                    line_list = []
                    coord_list = np.array([[]])
                    #Increment number of regions
                    region_count +=1

            line_list.append(line)
            if bboxes_line_dict[line].size != 0:
    
                if coord_list.size == 0:
                    #Rotate angle in coords list back to original for bbox purposes
                    coord_list = bboxes_line_dict[line]
                else:
                    #Rotate angle in coords list back to original for bbox purposes
                    coord_list = np.concatenate([coord_list, bboxes_line_dict[line]])
            region_dict[region_count]["lines"] = line_list

            if line == len(bboxes_line_dict):
                #Find bounding box that encompasses all the lines within a region
                #Unrotate coord_list
                rotation_matrix = np.array([[np.cos(baseline_angle_data), np.sin(baseline_angle_data)],[-np.sin(baseline_angle_data),np.cos(baseline_angle_data)]])
                coord_list = np.matmul(coord_list, rotation_matrix)
                hull = ConvexHull(coord_list)
                vertices = hull.vertices
                coords_region = coord_list[vertices]
                coords_region = np.round(coords_region).astype(int)
                coords_region = ' '.join(map(str, coords_region.tolist()))
                coords_region=coords_region.replace(", ", ",")
                coords_region=coords_region.replace("[","")
                coords_region=coords_region.replace("]","") 
                region_dict[region_count]["coords"]  = coords_region
            
            #Set the value of y1 for next iteration comparisons by finding greatest y-value in convex hull of line
            # Make sure the line contains glyphs
            if len(bboxes_line_dict[line]) != 1:
                y1 = np.max(bboxes_line_dict[line][:, 1])
        return region_dict
    #If only one region
    else:
        for line in bboxes_line_dict:
            line_list.append(line)
        region_dict[1]["lines"] = line_list
        region_dict[1]["coords"] = bbox_for_group(line_transcript, transcript_list, "TextRegion", ref_txt_file, baseline_angle_data=0)
        return region_dict


'''
Create a bboxlist
Args:
- line (string): the text line with the bounding boxes 
- toString (boolean): determines if the bbox list is in string format or not
Returns:
- bbox_list (string): list of all the coordinates in the line
'''
def create_bbox_list(line, toString):
    bbox_list = [] 
    #Adds bbox coordinates as floats    
    bbox_list.extend([round(float(n)) for n in line.split()])
    #Return not as string but as coords
    if toString == False:
        bbox_array = np.array([[bbox_list[0], bbox_list[1]], [bbox_list[2], bbox_list[3]], [bbox_list[4], bbox_list[5]], [bbox_list[6], bbox_list[7]]])
        return bbox_array
    else:
        #Strips brackets in list and converts to string
        bbox_list = str(bbox_list)
        bbox_list=bbox_list.replace("[","")
        bbox_list=bbox_list.replace("]","") 
        #Replaces the commas in the bbox to get the correct format
        for x in range(2,5):
            bbox_list = replace_nth(",","",bbox_list,x)
        #Corrects the spacing for bbox_list
        for x in range (1,5):
            bbox_list = replace_nth(" ", "", bbox_list, x)
        return bbox_list


'''
Create a bounding box for text regions and text lines
Args:
- line_transcript (int): the line where the transcript begins
- transcript_list (list): the list of all characters in the transcript
- key (string): determine if bounding box is being made for a text region or a text line
- ref_txt_file (file): the .txt file being iterated through
Returns:
- either dictionary of convex hull bboxes for a line or bbox for a text region
'''
def bbox_for_group(line_transcript, transcript_list, key, ref_txt_file, baseline_angle_data=0):
    #Reset to beginning of file
    ref_txt_file.seek(0)

    transcript_list_i = 0
    line_number = 1
    
    bboxes_arr = np.array([[]])
    
    bboxes_line_dict ={}

    glyph_dict = {}

    baseline_angle_data = np.radians(float(baseline_angle_data))

    #How to multiply arrays: https://stackoverflow.com/questions/71883631/how-to-perform-matrix-multiplication-in-python
    rotation_matrix = np.array([[np.cos(baseline_angle_data), -np.sin(baseline_angle_data)],[np.sin(baseline_angle_data),np.cos(baseline_angle_data)]])

    for i, line in enumerate(ref_txt_file):
        if i>= 3 and i<line_transcript-1: 
            if key == "Glyph":
                bbox_list = create_bbox_list(line, True)
                glyph_dict[i-2] = bbox_list

            elif key == "TextRegion": #Call bbox method for a range of lines (determine where i is for line 1-2 for instance)

                #Calls bbox method for each line
                bbox_list = create_bbox_list(line, False)
                #To prevent space at start of coords
                if i == 3:
                    bboxes_arr = bbox_list
                else:
                    bboxes_arr = np.concatenate([bboxes_arr, bbox_list])
                    
            #To determine lines in string or bbox format
            else:
                #Calls bbox method for each line to convert to correct format
                bbox_list = create_bbox_list(line, False)

                if transcript_list[transcript_list_i] == "$":
                    if len(bboxes_arr) != 1:
                        hull = ConvexHull(bboxes_arr)
                        vertices = hull.vertices
                        coords_region = bboxes_arr[vertices]
                        #If lines should not be converted to a string (NStr)
                        if key == "TextLineNStr":
                            result = np.matmul(coords_region, rotation_matrix)
                            bboxes_line_dict[line_number] = result
                        else:
                            coords_region = ' '.join(map(str, coords_region.tolist()))
                            coords_region=coords_region.replace(", ", ",")
                            coords_region=coords_region.replace("[","")
                            coords_region=coords_region.replace("]","") 
                            #All the bboxes for a line
                            bboxes_line_dict[line_number] = coords_region
                        bboxes_arr = np.array([[]])
                        line_number += 1
                        transcript_list_i += 1
                    #if first line of file is new line, do I want to have empty lines? There is a line in the transcript but no data...
                    else:
                        bboxes_line_dict[line_number] = bboxes_arr
                        line_number += 1
                        transcript_list_i += 1
                #Bbox list for first letter is still generated, so it is used below
                if i == 3 or transcript_list[transcript_list_i-1] == "$":
                    bboxes_arr = bbox_list
                else:
                    bboxes_arr = np.concatenate([bboxes_arr, bbox_list])
                
                transcript_list_i += 1
                #If end of file (issue because two last chars are new lines not one)
                if (i == line_transcript-2):
                    if len(bboxes_arr) != 1:
                        hull = ConvexHull(bboxes_arr)
                        vertices = hull.vertices
                        coords_region = bboxes_arr[vertices]
                        #Convert to cleaned string
                        if key == "TextLineNStr":
                            result = np.matmul(coords_region, rotation_matrix)
                            bboxes_line_dict[line_number] = result
                        else: 
                            coords_region = ' '.join(map(str, coords_region.tolist()))
                            coords_region=coords_region.replace(", ", ",")
                            coords_region=coords_region.replace("[","")
                            coords_region=coords_region.replace("]","") 
                            #All the bboxes for a line
                            bboxes_line_dict[line_number] = coords_region
                    else:
                        #In case line has no valid glyphs (all *)
                        bboxes_line_dict[line_number] = bboxes_arr
    #Go to start of file for next use
    ref_txt_file.seek(0)
    if key == "Glyph":
        return glyph_dict
    elif key == "TextRegion":
        hull = ConvexHull(bboxes_arr)
        vertices = hull.vertices
        coords_region = bboxes_arr[vertices]
        coords_region = ' '.join(map(str, coords_region.tolist()))
        coords_region=coords_region.replace(", ", ",")
        coords_region=coords_region.replace("[","")
        coords_region=coords_region.replace("]","") 
        return coords_region
    else:
        return bboxes_line_dict

'''
Create the the TextRegions for each line in the file
Args:
- ref_txt_file (file): the original text file with information
- line_transcript (int): the line where the transcript begins
- transcript_list (list): list of all letters in the transcript
- OrderedGroup (xml section)
- Page (xml section)
'''
def create_text_regions(ref_txt_file, line_transcript, transcript_list, Page, baseline_angle_data):
    #Records transcript list index
    transcript_list_i = 0

    glyph_counter = 1

    #Dictionary for lines and bboxes
    bboxes_line_dict = bbox_for_group(line_transcript, transcript_list, "TextLine", ref_txt_file)
    #Create ReadingOrder group
    ReadingOrder=ET.SubElement(Page, "ReadingOrder")
    OrderedGroup=ET.SubElement(ReadingOrder, "OrderedGroup")
    OrderedGroup.attrib["id"] = "r"+str(baseline_angle_data)

    #Region dict has regions as keys and line number as values for that region
    region_dict = find_regions(line_transcript, transcript_list, ref_txt_file, baseline_angle_data)
    #Get dictionary of glyph coordinates
    glyph_dict = bbox_for_group(line_transcript, transcript_list, "Glyph", ref_txt_file)

    for region in region_dict:
        #Reading order attribute
        RegionRefIndexed = ET.SubElement(OrderedGroup,"RegionRefIndexed")
        RegionRefIndexed.attrib["regionRef"] = "r" + str(region)
        RegionRefIndexed.attrib["index"] = str(region)
        #Creates text region with id and type
        TextRegion = ET.SubElement(Page, "TextRegion")
        TextRegion.attrib["id"] = "r" + str(region)
        #Assign type as paragraph
        TextRegion.attrib["type"] = "paragraph"
        TextRegionCoords = ET.SubElement(TextRegion, "Coords")
        #Coords for all glyphs in TextRegion (for the entire file when we want select areas)
        TextRegionCoords.attrib["points"] = region_dict[region]["coords"]

        for line in region_dict[region]["lines"]:
            #If glyph contents in a line
            if len(bboxes_line_dict[line]) != 1:
                TextLine = ET.SubElement(TextRegion, "TextLine")
                TextLine.attrib["id"] = "l" + str(line)
                TextLineCoords = ET.SubElement(TextLine, "Coords")
                TextLineCoords.attrib["points"] = bboxes_line_dict[line]
                #Wrap Glyph in word so schema works correctly (word is entire line)
                Word=ET.SubElement(TextLine, "Word")
                Word.attrib["id"] = "w" + str(line) 
                WordCoords = ET.SubElement(Word, "Coords")
                WordCoords.attrib["points"] = bboxes_line_dict[line]
                #Extract all the glyphs in each line
                while transcript_list_i < len(transcript_list) and transcript_list[transcript_list_i] != "$":
                    Glyph=ET.SubElement(Word, "Glyph")
                    Glyph.attrib["id"] = "g" + str(glyph_counter) 
                    GlyphCoords = ET.SubElement(Glyph, "Coords")
                    GlyphCoords.attrib["points"] = glyph_dict[glyph_counter]
                    glyph_counter += 1
                    TextEquiv = ET.SubElement(Glyph, "TextEquiv")
                    ET.SubElement(TextEquiv, "Unicode").text=transcript_list[transcript_list_i]
                    transcript_list_i += 1
                #Increment if it equals new line
                transcript_list_i +=1
            #If line has no glyph info, its just a new line char
            else:
                transcript_list_i += 1
'''
Makes one PageXML file
Args:
- xml_storage_directory: Path to the folder where created .xml files are stored.
- directory_txt_files: Path to the folder where letter box .txt files are stored.
- directory_png_files: Path to the folder where letter box .png images are stored.
- filename: The name of the letter box .txt file to convert to .xml.
- is_greek: boolean for whether characters stored in XML are Greek or Latin.
'''      
def make_one_xml(filename, directory_txt_files, directory_png_files, xml_storage_directory, is_greek):
    #To verify file is being created
    print(filename)
    
    #The reference txt file
    ref_txt_file_name = os.path.join(directory_txt_files, filename)

    #path to xml files with the filename included
    xml_file_path = os.path.join(xml_storage_directory, filename.strip(".txt") + ".xml")

    #Create PcGts section and validate against xsd schema
    PcGts = ET.Element('PcGts')
    PcGts.attrib["xmlns"] ="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
    PcGts.attrib["xmlns:xsi"] = "http://www.w3.org/2001/XMLSchema-instance"
    PcGts.attrib["xsi:schemaLocation"] = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd"

    #Adds MetaData section
    Metadata = ET.SubElement(PcGts, "Metadata")

    #Adds xml file creator
    ET.SubElement(Metadata, "Creator").text="Maura"

    #https://stackoverflow.com/questions/10607688/how-to-create-a-file-name-with-the-current-date-time-in-python
    current_day = str(date.today()) + "T" + str(datetime.now().strftime("%H:%M:%S"))

    #Adds created data (current day/time)
    Created = ET.SubElement(Metadata, "Created")
    Created.text = current_day

    #Adds last change (also current day/time)
    ET.SubElement(Metadata, "LastChange").text=str(current_day)

    #Image png file name is added
    image = filename.replace("_letters.txt",".png")

    #Adds Page section to xml
    Page = ET.SubElement(PcGts, "Page")
    Page.attrib["imageFilename"] = image

    #Checks to see if png file exists in png_images folder and sets height/width (https://stackoverflow.com/questions/1507084/how-to-check-dimensions-of-all-images-in-a-directory-using-python)
    imagename = os.path.join(directory_png_files,image)
    try:
        im = Image.open(imagename)
        width, height = im.size
        Page.attrib["imageWidth"] = str(width)
        Page.attrib["imageHeight"] = str(height)
    except(FileNotFoundError):
        print("\nNo image for this file has been found in the given directory. Refer to the instructions below for help with calling in the command line.\n")
        calling_instructions()
    #Determines where the transcript line begins and, thus, where transcript appears in file
    line_transcript = word_appears(ref_txt_file_name, "Transcript")

    #reads baseline angle
    baseline_angle_data = read_line(ref_txt_file_name, 1, True)
    #creates transcript list
    transcript_list = create_transcript_list(ref_txt_file_name, line_transcript, is_greek)

    #reads old file from the beginning
    with open(ref_txt_file_name, 'r') as ref_txt_file:
        
        #Create the different text regions in the file
        create_text_regions(ref_txt_file, line_transcript, transcript_list, Page, baseline_angle_data)

        #Writes to xml file
        tree = ET.ElementTree(PcGts)
        #Provided pretty print information (https://stackoverflow.com/questions/28813876/how-do-i-get-pythons-elementtree-to-pretty-print-to-an-xml-file)
        ET.indent(tree, space="\t", level=0)
        tree.write(xml_file_path, xml_declaration=True, encoding='utf-8')

'''
Makes all the PageXML files
Args:
- xml_storage_directory: Path to the folder where created .xml files are stored.
- directory_txt_files: Path to the folder where letter box .txt files are stored.
- directory_png_files: Path to the folder where letter box .png images are stored.
- is_greek: boolean for whether characters stored in XML are Greek or Latin.
'''      
def make_all_xml(directory_txt_files, directory_png_files, xml_storage_directory, is_greek):
    # iterate over files in original letter box text files
    try:
        for filename in os.listdir(directory_txt_files):
            make_one_xml(filename, directory_txt_files, directory_png_files, xml_storage_directory, is_greek)
    except(FileNotFoundError):
        print("\nYour file or given directory does not exist! Refer to the instructions below for help with calling in the command line.\n")
        calling_instructions()


#How to make command line args: https://stackoverflow.com/questions/3987041/run-function-from-the-command-line
if __name__ == "__main__":
    #(https://docs.python.org/2/howto/argparse.html)
    parser = argparse.ArgumentParser(
    #How to format: https://stackoverflow.com/questions/50021282/python-argparse-how-can-i-add-text-to-the-default-help-message
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=textwrap.dedent('''
    INSTRUCTIONS FOR SETUP:
        1. Download a folder with letter box .txt files.
        2. Download a folder with .png images for the respective .txt files.
        3. You are now ready to call in the command line:
            POSITIONAL ARGUMENTS:
            a. To create one XML file from a directory: python3 make_xml.py FILENAME.txt *LETTER_BOXES_DIRECTORY *PNG_IMAGES_DIRECTORY *XML_STORAGE_DIRECTORY *IS_GREEK
            b. To create all XML files from a directory: python3 make_xml.py *LETTER_BOXES_DIRECTORY *PNG_IMAGES_DIRECTORY *XML_STORAGE_DIRECTORY *IS_GREEK
            KEYWORD ARGUMENTS (arguments can be in any order):
            c. To create one XML file from a directory: python3 make_xml.py -s FILENAME.txt -o *LETTER_BOXES_DIRECTORY -im *PNG_IMAGES_DIRECTORY -xml *XML_STORAGE_DIRECTORY -greek *IS_GREEK
            d. To create all XML files from a directory: python3 make_xml.py -o *LETTER_BOXES_DIRECTORY -im *PNG_IMAGES_DIRECTORY -xml *XML_STORAGE_DIRECTORY -greek *IS_GREEK
            BLEND KEYWORD/POSITIONAL ARGUMENTS:
            e. You can also make calls with positional arguments followed by keyword arguments (ex: 'python3 make_xml.py FILENAME.txt -im *PNG_IMAGES_DIRECTORY' or 'python3 make_xml.py FILENAME.txt *LETTER_BOXES_DIRECTORY -xml *XML_STORAGE_DIRECTORY -im *PNG_IMAGES_DIRECTORY'
                - In these cases, the arguments without keywords must be in the order they appear (ex: FILENAME.TXT then LETTER_BOXES_DIRECTORY, etc.) and must appear BEFORE the keyword arguments.
                XML_STORAGE DIRECTORY: Path to the folder where you would like to store your created .xml files.
                LETTER_BOXES_DIRECTORY: Path to the folder where you have stored your letter box .txt files.
                PNG_IMAGES_DIRECTORY: Path to the folder where you have stored your .png images.
                FILENAME: The name of the letter box .txt file you would like to convert to .xml.
                IS_GREEK: Boolean for whether characters stored in XML are Greek or Latin. Type (not case sensitive) "t" or "f".
            - Note: The directory may end with or without "/" or "\." Ex: Users/mp/Squeezes/letter_boxes or Users/mp/Squeezes/letter_boxes/
            - *Optional: defaults to current working directory (or false for IS_GREEK) if no argument provided.
    COMMON UNRECOGNIZED ARGUMENT ERRORS:
        - Mispelled or nonexistent file path
        - Writing .xml rather than .txt when including filename arg
    '''
    )
    )
    #For keyword arguments
    #For a single text file
    parser.add_argument('-s', type=str)
    #for a text directory
    parser.add_argument('-o', type=str)
    parser.add_argument('-im', type=str)
    parser.add_argument('-xml', type=str)
    parser.add_argument('-greek', type=str)

    #For positional arguments
    parser.add_argument('args', nargs='*')
    args = parser.parse_args()

    if len(args.args) == 0:
        print("\nYou entered no arguments! At least one argument is required.\n")
        calling_instructions()
        quit()

    if args.s == None and args.args:
        if ".txt" in args.args[0]:
            args.s = args.args[0]
            args.args.pop(0)
    
    if args.o == None:
        if args.args:
            args.o = args.args[0]
            args.args.pop(0)
        else:
            args.o = os.getcwd()
    
    if args.im == None:
        if args.args:
            args.im = args.args[0]
            args.args.pop(0)
        else:
            args.im = os.getcwd()
    
    if args.xml == None:
        if args.args:
            args.xml = args.args[0]
            args.args.pop(0)
        else:
            args.xml = os.getcwd()

    if args.greek == None:
        if args.args:
            args.greek = args.args[0]
            args.args.pop(0)
        else:
            args.greek = True

    #Convert to boolean
    if args.greek == "f" or args.greek == "F" or args.greek == False:
        args.greek = False
    elif args.greek == "t" or args.greek == "T" or args.greek == True:
        args.greek == True
    else:
        #if true or false not provided
        print("Must provided True or False for -greek argument!")
        quit()
    
    if args.s != None:
        make_one_xml(args.s, args.o, args.im, args.xml, args.greek)
        exit()
    else:
        make_all_xml(args.o, args.im, args.xml, args.greek)
        exit()

    # #Catches if no args provided or file not found: https://stackoverflow.com/questions/14016742/detect-and-print-if-no-command-line-argument-is-provided
    # except (FileNotFoundError):  #FileNotFoundError, IndexError, TypeError
