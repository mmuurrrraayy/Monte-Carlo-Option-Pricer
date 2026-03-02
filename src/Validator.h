#ifndef Validator_H
#define Validator_H
#include <iostream>

class Validator{
    public:

        //checks positive integer
        static void is_valid_positive_integer(int check){
             try
                {
                    if(check>0){}
                       return;
                }
                catch(const std::exception& e)
                {   
                    std::cout << "Invalid variable" << check<< std::endl;
                
                }
        }

        //checks positive doubble
        static void is_valid_positive_double(double check){
             try
                {
                    if(check>0.0){}
                       return;
                }
                catch(const std::exception& e)
                {   
                    std::cout << "Invalid variable" << check<< std::endl;
                
                }
        }

};

#endif

