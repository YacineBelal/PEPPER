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
        noeud3: Node {
            @display("p=207,141");
        }
        noeud4: Node {
            @display("p=260,141");
        }
        noeud5: Node {
            @display("p=319,141");
        }
        noeud6: Node {
            @display("p=378,141");
        }
        noeud7: Node {
            @display("p=437,141");
        }
        noeud8: Node {
            @display("p=494,141");
        }
        noeud9: Node {
            @display("p=551,141");
        }
        noeud10: Node {

            @display("p=613,141");
        }
    connections:
        server.sl++ <--> {  delay = 100ms; } <--> noeud1.nl++;
        server.sl++ <--> {  delay = 100ms; } <--> noeud2.nl++;
        server.sl++ <--> {  delay = 100ms; } <--> noeud3.nl++;
        server.sl++ <--> {  delay = 100ms; } <--> noeud4.nl++;
        server.sl++ <--> {  delay = 100ms; } <--> noeud5.nl++;
        server.sl++ <--> {  delay = 100ms; } <--> noeud6.nl++;
        server.sl++ <--> {  delay = 100ms; } <--> noeud7.nl++;
        server.sl++ <--> {  delay = 100ms; } <--> noeud8.nl++;
        server.sl++ <--> {  delay = 100ms; } <--> noeud9.nl++;
        server.sl++ <--> {  delay = 100ms; } <--> noeud10.nl++;


}
