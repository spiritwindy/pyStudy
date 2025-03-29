# temp = "5 degrees"

# cel = 0

# try:

#     fahr = float(temp)

#     cel = (fahr - 32.0) * 5.0 / 9.0

# except:

#     print("test1.py")
# print(cel)


def jls_extract_def():
    
    smallest = None
    print("Before:", smallest)
    for itervar in [3, 41, 12, 9, 74, 15]:
        if smallest is None or itervar < smallest:
            smallest = itervar
            # break
            continue
        print("Loop:", itervar, smallest)
    print("Smallest:", smallest)
    return smallest


# smallest = jls_extract_def()


# for n in "banana":
#     print(n)

# words = 'His e-mail is q-lar@freecodecamp.org'
# pieces = words.split()
# print(pieces)
# parts = pieces[3].split('-')
# n = parts[1]


counts = { 'quincy' : 1 , 'mrugesh' : 42, 'beau': 100, '0': 10}


# lst = []
# for key, val in counts.items():
#     newtup = (val, key)
#     lst.append(newtup)
# lst = sorted(lst, reverse=True)
# print(lst)


# import urllib.request
# fhand = urllib.request.urlopen('http://data.pr4e.org/romeo.txt')
# for line in fhand:
#     print(line.decode().strip())


class PartyAnimal:
    x = 0
    def party(self):
        self.x = self.x + 2
        print(self.x)

an = PartyAnimal()
an.party()
an.party()
print(an.x)