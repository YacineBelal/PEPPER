package asynchGL;
simple Server
{
   gates:
       inout sl[];

}
simple Node
{
    gates:
        inout no[100];
        inout nl[];

}

network RandomGraph
{
    parameters:
        int size = default(100);
        
    submodules: 
         server: Server;
         Noeuds[size] : Node;

    connections allowunconnected:
        for i = 0 .. size - 1{ 
        server.sl++ <--> {} <--> Noeuds[i].nl++;
        }
        for i = 0 .. size - 1,for k = i + 1  .. size - 1{
        Noeuds[i].no++ <--> {} <--> Noeuds[k].no++;     
        }
           
}