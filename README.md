Folder-ul Landscapes contine 7 imagini cu munti alese cat mai simplu pentru a-i face GAN-ului treaba mai usoara.
Dupa ce rulez script-ul image_proc.py imi va face resize la fiecare imagine din folder-ul Landscapes, mai exact va modifica fiecare imagine la o rezolutie 64x64 si le va pune in Landscapes1.
Din Landscapes1 imaginile vor fi folosite pentru antrenarea GAN-ului
Dupa rularea codului, gan_train.py, care foloseste ca motor de functionare un GAN ar trebui sa genereze imagini similare cu cele din Landscapes1
*Imaginile trebuie ajustate la o rezolutie mai mica pentru a fi mai eficienta antrenarea GAN-ului.
