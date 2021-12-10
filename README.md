Readme

Tentang program:
Program ini adalah program untuk mendapatkan data jalan menggunakan data citra streetmap google imagery. Input program ini adalah sebuah citra dengan format penamaan: 
x_y_z.png, dimana x merupakan koordinat tiles x, y merupakan koordinat tiles y, dan z merupakan level zoom citra tsb. Asumsi citra berukuran standard 256x256 piksel. 
Output dari program ini adalah :
1. List koordinat points jalan yang berhasil dideteksi program sebagai returns dari fungsi utama DeteksiJalanStreetMap Function
2. (Opsional) List Gambar yang menggambarkan proses yang dilakukan dalam program, mulai dari pembentukan mask image, deteksi kontur, delauney triangulation, voronoi diagram dan result points
3. (Opsional) SHP file yang berisi points dan lines yang berhasil terdeteksi.

Petunjuk penggunaan:

