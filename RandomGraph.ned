simple Service
{
    gates:
        inout sl[];

}

simple Node
{
    gates:

        inout nl[1];
        inout no[600];

}

network RandomGraph
{
    @display("bgb=835,197");
    submodules:
        service : Service{
            @display("p=339,24");
        }
         Noeuds[600] : Node;
    connections allowunconnected:
        for i = 0 .. 599{ 
        service.sl++ <--> <--> Noeuds[i].nl++;
        }
        for i = 0 .. 598, for j=i+1 .. 599{
        Noeuds[i].no[j] <--> <--> Noeuds[j].no[i] if uniform(0,1) < 0.20;      
        }
        





}
