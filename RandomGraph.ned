simple Node
{
    gates:
        inout no[100];

}

network RandomGraph
{
    parameters:
        int size = default(100);
        
    submodules: 
         Noeuds[size] : Node;

    connections allowunconnected:
        for i = 0 .. size - 1, for j= 0 .. 6{
        Noeuds[i].no[intuniform(0,size-1)] <--> {@reconnect;} <--> Noeuds[intuniform(0,size-1)].no[intuniform(0,size-1)];     
        }
        
}
