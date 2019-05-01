package mx.dxtr.dogs

import android.support.v7.app.AppCompatActivity
import android.os.Bundle
import android.content.Intent
import android.util.Log
import android.view.View
import android.widget.Toast
import es.dmoral.toasty.Toasty
import org.json.JSONObject
import java.util.*


class MainActivity : AppCompatActivity() {

    var token : String = ""
    var first_name : String = ""
    var last_name : String = ""
    var email : String = ""
    var id : String = ""
    var birthday : String = ""

    fun login(view : View) {
        val intent = Intent(this, PhotoActivity::class.java)
        startActivity(intent)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
    }

}
