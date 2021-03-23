simple Server
{
    gates:
        inout sl[];

}

simple Node
{
    gates:

        inout nl[];

}

network PyTictoc
{
    @display("bgb=835,197");
    submodules:
        server: Server {
            @display("p=339,24");
        }
         Noeuds[943] : Node;
    connections:
        for i = 0 .. 942{ 
        server.sl++ <--> <--> Noeuds[i].nl++;
        }

}
