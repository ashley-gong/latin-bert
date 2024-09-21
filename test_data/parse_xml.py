import xml.etree.ElementTree as ET

def parseXML(file):
  tree = ET.parse(file)
  root = tree.getroot()

  for item in root.findall('.//div2'):
    firstLine = item.find('l')
    poemNumber = item.get('n')
    if firstLine is not None:
      lineText = ''.join(firstLine.itertext())
      for punc in '.,;:!?':
        lineText = lineText.replace(punc, '')
      if lineText:
        print(poemNumber, lineText)

# python3 parse_xml.py > output.txt
if __name__ == "__main__":
  with open("catullus_test.xml") as file:
    parseXML(file)  
