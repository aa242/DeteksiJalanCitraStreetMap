**Readme

**Tentang program:
Program ini adalah program untuk mendapatkan data jalan menggunakan data citra streetmap google imagery. Input program ini adalah sebuah citra dengan format penamaan: 
x_y_z.png, dimana x merupakan koordinat tiles x, y merupakan koordinat tiles y, dan z merupakan level zoom citra tsb. Asumsi citra berukuran standard 256x256 piksel. 
Output dari program ini adalah :
1. List koordinat points jalan yang berhasil dideteksi program sebagai returns dari fungsi utama DeteksiJalanStreetMap Function
2. (Opsional) List Gambar yang menggambarkan proses yang dilakukan dalam program, mulai dari pembentukan mask image, deteksi kontur, delauney triangulation, voronoi diagram dan result points
3. (Opsional) SHP file yang berisi points dan lines yang berhasil terdeteksi.

**Petunjuk penggunaan:

Fungsi utama program ini terdapat pada file MainFunc_DeteksiJalanStreetMap.py, dengan fungsi utama :
DeteksiJalanStreetMap(filename, outputPointsOnly=False, outputImages=False, outputSHP=False)

dengan input:
1. filename = nama input image streetmap, dengan format yang ditentukan sebelumnya
2. outputPointsOnly = jika False, maka akan mengoutputkan tidak hanya list of points tapi juga list of lines
3. outputImages = jika False, maka tidak akan mengoutputkan images progress pengolahan, spt hasil delauney triangulation
4. outputSHP = jika False, maka tidak akan mengoutputkan file shp, misal untuk menghemat memory

**Contoh penggunaan:

Contoh penggunaan untuk data citra streetmap seluruh cimahi (di folder /data_Cimahi), diberikan sebagai contoh pada script TestFunc_DeteksiJalanStreetMap.py, dimana
contoh penggunaanya untuk pemanggilan fungsi DeteksiImageStreetMap, sbb:

```
returned_points, returned_lines = DeteksiJalanStreetMap(filename, outputPointsOnly=False, outputImages=False, outputSHP=False)
```

dimana untuk mengoutputkan hasilnya (berupa points, atau points dan lines, sbb:

```
from HelperFuncs_Output import *

flat_list_points = []
for sublist in list_all_points_area:
    for item in sublist:
        flat_list_points.append(item)
            

flat_list_edges = []
for sublist in list_all_lines_area:
    for item in sublist:
        flat_list_edges.append(item)            


fileout = 'Cimahi_Deteksi.png'

Output_SHP_Points(flat_list_points, fileout)
    
Output_SHP_Lines(flat_list_edges, fileout)
```

**Gambaran tahapan pengolahan program:

1. Mask Detection

![alt text](https://github.com/aa242/DeteksiJalanCitraStreetMap/blob/master/maskimage_streetmap_detJalan.jpg?raw=true)

2. Contour Detection

![alt text](https://github.com/aa242/DeteksiJalanCitraStreetMap/blob/master/countours1_streetmap_detJalan.jpg?raw=true)
![alt text](https://github.com/aa242/DeteksiJalanCitraStreetMap/blob/master/countours2_streetmap_detJalan.jpg?raw=true)
![alt text](https://github.com/aa242/DeteksiJalanCitraStreetMap/blob/master/countours3_0_streetmap_detJalan.jpg?raw=true)
![alt text](https://github.com/aa242/DeteksiJalanCitraStreetMap/blob/master/countours4_streetmap_detJalan.jpg?raw=true)


3. Delauney Triangulation


![alt text](https://github.com/aa242/DeteksiJalanCitraStreetMap/blob/master/delauney_streetmap_detJalan.jpg?raw=true)
![alt text](https://github.com/aa242/DeteksiJalanCitraStreetMap/blob/master/delauneyImage_streetmap_detJalan.jpg?raw=true)

4. Voronoi Diagram

![alt text](https://github.com/aa242/DeteksiJalanCitraStreetMap/blob/master/voronoi_streetmap_detJalan.jpg?raw=true)
![alt text](https://github.com/aa242/DeteksiJalanCitraStreetMap/blob/master/voronoiImage_streetmap_detJalan.jpg?raw=true)

5. Points and Lines Detection

![alt text](https://github.com/aa242/DeteksiJalanCitraStreetMap/blob/master/pointsJalan_streetmap_detJalan.jpg?raw=true)
![alt text](https://github.com/aa242/DeteksiJalanCitraStreetMap/blob/master/linesJalan_streetmap_detJalan.jpg?raw=true)

Example Result untuk kota Cimahi:

