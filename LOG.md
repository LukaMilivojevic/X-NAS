# DNEVNI LOG #

### 16.08.2020. ###

* Pocet rad na NSGA-II genetskom algoritmu. Za sada je potrebno realizovati 
  ga na prostom primeru optimizacije bit-distance u odnosu na 2 razlicita niza,
  a kasnije ce biti modifikovan i prilagodjen radu sa enkodovanim neuronskim mrezama.

* Napravljene funkcije za generisanje resenja, racunanje fitness i generisanje populacije.

* Dodate funkcije za pronalazenje prvog pareto fronta i plotovanje jedinki i fitnesa.

### 17.08.2020. ###

* Dodato pronalazenje i ostalih frontova, ali potrebno je jedinke iz njih birati crowding distance-om kad ne mogu da stanu.
  Trenutno u front, kada ne staje ceo u sledecu generaciju, upada prva jedinka koja se pronadje u njemu.

* Imao sam solidno problema sa ovim delom pa je zato usporeno.


### 18.08.2020. ###

* Grafici pre i posle selekcije jedinki:
![alt text](https://bitbucket.org/pfepetnica/x_nas/src/nsga/NSGA-II/Pocetna%20populacija.png)
![alt text](https://bitbucket.org/pfepetnica/x_nas/src/nsga/NSGA-II/Nakon%20selekcije.png)


### 19.08.2020. ###

* Radi se na zavrsavanju citavog genetskog algoritma. Jedna greska deli od zavrsetka (nadam se makar :) )


### 20.08.2020. - 23.08.2020. ###

* Bio na Zlatiboru, nista nisam radio prakticno.

### 24.08.2020. ###

* Zavrsen genetski algoritam i dodata jos jedna optimizacija cisto da se lepse prikaze :).

### 25.08.2020. ###

* Dodata mutacija u algoritam.

### 26-29.08.2020. ###

* Zavrsena prva verzija kompletnog algoritma sa neuronskim mrezama, sada se moze preci na novi tip enkodovanja. 

* Dosta vremena potroseno na treniranje algoritma jer je potrebno oko 8 sati za 50 jedinki u 50 generacija.

* Kada se zavrsi sve bice utroseno i vise (npr. dan) kako bi se probalo sa vise jedinki u vise generacija.

* Pocetna populacija: ![alt text](https://bitbucket.org/pfepetnica/x_nas/src/networks/generacija0.png)

* Krajnja populacija: ![alt text](https://bitbucket.org/pfepetnica/x_nas/src/networks/poslednja.png)

### 30.08.-04.09.2020. ###

* Treniranja algoritma i sastanci vezani za moguca unapredjenja sem novog enkodovanja.

### 05.09.2020. ###

* Pocet rad na novom tipu enkodovanja i izvestaju.