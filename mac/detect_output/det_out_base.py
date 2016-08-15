import abc

class DetOutBase(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self):
        self.name= "DetOutBase"
        
    def write_event(self,event_data):
        self.__write_event__(self,event_data)
    
    @abc.abstractmethod
    def __write_event__(self,event_data):
        """
        I guess event_data will be dictionary of arbitrary info you can send to child class
        """
        
                    
        
        
    @abc.abstractmethod
    def __close__(self):
        """
        finalize readout
        """
