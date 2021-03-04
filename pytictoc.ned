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
    @display("bgb=758,197");
    submodules:
        server: Server {
            @display("p=339,24");
        }
        noeud1: Node {
            @display("p=101,141");
        }
        noeud2: Node {
            @display("p=151,141");
        }
    connections:
        server.sl++ <--> {  delay = 100ms; } <--> noeud1.nl++;
        server.sl++ <--> {  delay = 100ms; } <--> noeud2.nl++;
        
}
