#include <stdlib.h>
#include <unistd.h>

int main(){

  while(1){
    char * space = (char *) malloc(200 * 1024 * 1024);
    sleep(10);
    free(space);
    sleep(10);
  }
  return 0;
}
