import pywaspgen.filters as filters
import pywaspgen.modems.modem as modem

class DIGITAL(modem.MODEM):
    """
    Digital modem base class.
    """
    def __init__(self, sig_type, pulse_type):
        """
        The constructor for the base `DIGITAL` class.

        Args:
            sig_type (dict): The signal type of the digital modem.
            pulse_type (dict): The pulse shape metadata of the digital modem.
        """
        super().__init__(sig_type, pulse_type)
        self._symbol_table_create()
        self.__set_pulse_shape()

    def _symbol_table_create(self):
        """
        Creates the modem's IQ data symbol table of the type specified by ``DIGITAL.sig_type``.
        """        
        self.__symbol_table_create()  

    def __set_pulse_shape(self):
        """
        Sets the pulse shaping filter of the modem based on ``DIGITAL.pulse_type``.
        """
        class_method = getattr(filters, self.pulse_type["format"])
        self.pulse_shaper = class_method(self.pulse_type)

    def set_sps(self, sps):
        """
        Sets the samples per symbol of the modem.

        Args:
            sps (float): The samples per symbol to be used by the modem when performing filtering operations.
        """
        self.pulse_type["sps"] = sps
        self.__set_pulse_shape(self)  