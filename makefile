include ${OCCA_DIR}/scripts/makefile

flags += -DtFloat=float -DOCCA_GL_ENABLED=1 -Dp_N=$(N) -DNDG3d -g 
ifeq ($(OS),OSX)
	links += -framework OpenGL -framework GLUT
endif

ifeq ($(OS),LINUX)
	links +=
# -lGLU -lglut
endif

oPath = ./obj
sPath = ./src
links += -L./lib  -lparmetis -lmetis

#---[ COMPILATION ]-------------------------------
headers = $(wildcard $(iPath)/*.hpp) $(wildcard $(iPath)/*.tpp)
sources = $(wildcard $(sPath)/*.cpp)

objects  = $(subst $(sPath)/,$(oPath)/,$(sources:.cpp=.o)) 

main: $(objects) $(headers) main.cpp
	$(compiler) $(compilerFlags) -o main $(flags) $(objects) main.cpp $(paths) $(links)

$(oPath)/%.o:$(sPath)/%.cpp $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.hpp))) $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.tpp)))
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

clean:
	rm -f $(oPath)/*;
	rm -f main;
#=================================================
